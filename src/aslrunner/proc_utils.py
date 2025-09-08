#!/usr/bin/env python3
import os, re, sys, time, struct, ctypes, ctypes.util, errno


def find_wine_process(name: str):
    """
    Find Wine processes whose cmdline contains a Windows-style path
    ending with '<name>.exe'. Returns a list of dicts with pid, exe_arg, and argv.
    """
    exe_pat = re.compile(rf"(?i)\b[A-Z]:\\.*\\{re.escape(name)}\.exe\b")

    def list_pids():
        for entry in os.listdir("/proc"):
            if entry.isdigit():
                yield int(entry)

    def read_cmdline(pid):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                parts = f.read().split(b"\x00")
            return [p.decode("utf-8", "ignore") for p in parts if p]
        except Exception:
            return []

    matches = []
    for pid in list_pids():
        argv = read_cmdline(pid)
        if not argv:
            continue
        if any(exe_pat.search(arg) for arg in argv):
            exe_arg = next(arg for arg in argv if exe_pat.search(arg))
            matches.append({"pid": pid, "exe_arg": exe_arg, "argv": argv})
    return matches


# -------- libc + process_vm_readv --------
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


class IOVec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]


# Configure libc.process_vm_readv if available
_has_pvr = hasattr(libc, "process_vm_readv")
if _has_pvr:
    libc.process_vm_readv.restype = ctypes.c_ssize_t
    libc.process_vm_readv.argtypes = [
        ctypes.c_int,  # pid_t
        ctypes.POINTER(IOVec),
        ctypes.c_ulong,  # liovcnt
        ctypes.POINTER(IOVec),
        ctypes.c_ulong,  # riovcnt
        ctypes.c_ulong,  # flags
    ]


def _process_vm_read(pid: int, addr: int, size: int) -> bytes:
    if size <= 0:
        return b""
    if not _has_pvr:
        raise OSError(errno.ENOSYS, "process_vm_readv not available in libc")

    buf = ctypes.create_string_buffer(size)
    local = IOVec(ctypes.cast(buf, ctypes.c_void_p), size)
    remote = IOVec(ctypes.c_void_p(addr), size)
    nread = libc.process_vm_readv(
        int(pid), ctypes.byref(local), 1, ctypes.byref(remote), 1, 0
    )
    if nread < 0:
        e = ctypes.get_errno()
        raise OSError(e, f"process_vm_readv failed: {os.strerror(e)}")
    return buf.raw[:nread]


# -------- maps parsing --------
def iter_maps(pid):
    with open(f"/proc/{pid}/maps", "r") as f:
        for line in f:
            rng, perms, offset, dev, inode, *path = line.strip().split()
            start_s, end_s = rng.split("-")
            start = int(start_s, 16)
            end = int(end_s, 16)
            path = path[0] if path else ""
            yield (start, end, perms, path)


def find_region(pid, addr, size=1):
    for start, end, perms, path in iter_maps(pid):
        if start <= addr and addr + size <= end:
            return (start, end, perms, path)
    return None


def get_module_base(pid, needle):
    if os.path.exists(needle):
        needle = os.path.basename(needle)
    needle = (needle or "").lower()
    for start, end, perms, path in iter_maps(pid):
        base = os.path.basename(path).lower()
        if needle and needle in base and ("r--p" in perms or "r-xp" in perms):
            return start
    return None


def get_all_modules(pid):
    modules = []
    for start, end, perms, path in iter_maps(pid):
        base = os.path.basename(path).lower()
        if ".exe" in base or ".dll" in base:
            modules.append({"start": start, "end": end, "perms": perms, "path": path})
    return modules


def get_module_memory(pid, needle):
    if os.path.exists(needle):
        needle = os.path.basename(needle)
    needle = needle.lower()
    for start, end, perms, path in iter_maps(pid):
        base = os.path.basename(path).lower()
        if needle == base and ("r--p" in perms or "r-xp" in perms):
            size = end - start
            try:
                return _process_vm_read(pid, start, size)
            except OSError:
                # Fall back to empty on failure; callers handle None/empty
                return None
    return None


# -------- safe remote read (process_vm_readv) --------
def pread_mem(pid: int, addr: int, size: int) -> bytes:
    return _process_vm_read(pid, addr, size)


# -------- pointer chain --------
def is_64_bit(pid):
    # Prefer the executable symlink for architecture detection
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
        with open(exe, "rb") as f:
            hdr = f.read(5)
            if hdr[:4] == b"\x7fELF":
                return hdr[4] == 2  # 2=ELFCLASS64, 1=ELFCLASS32
    except Exception:
        pass

    # Fallback: scan mapped executable regions
    for _, _, perms, path in iter_maps(pid):
        if "r-xp" in perms and path:
            try:
                with open(path, "rb") as f:
                    hdr = f.read(5)
                    if hdr[:4] == b"\x7fELF":
                        return hdr[4] == 2
            except Exception:
                continue

    # Last resort: assume current Python's pointer size
    return struct.calcsize("P") == 8


def read_ptr(pid, addr, ptr64=True):
    n = 8 if ptr64 else 4
    data = pread_mem(pid, addr, n)
    if len(data) != n:
        raise IOError("short read on pointer")
    return struct.unpack("<Q" if ptr64 else "<I", data)[0]


def follow_chain(pid, base_addr, offsets, ptr64=True):
    # first hop: read pointer at base_addr
    ptr = read_ptr(pid, base_addr, ptr64)
    if not offsets:
        return ptr
    for off in offsets[:-1]:
        ptr = read_ptr(pid, ptr + off, ptr64)
    return ptr + offsets[-1]


def read_cstring(pid, addr, maxlen=256):
    data = pread_mem(pid, addr, maxlen)
    return data.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")


def read_value(pid, addr, var_type: str):
    t = var_type.lower()
    if t.startswith("string"):
        # extract max length like string128
        try:
            maxlen = int(re.sub(r"[^0-9]", "", t))
            if maxlen <= 0:
                maxlen = 256
        except Exception:
            maxlen = 256
        return read_cstring(pid, addr, maxlen)
    if t in ("double",):
        data = pread_mem(pid, addr, 8)
        if len(data) != 8:
            raise IOError("short read on double")
        return struct.unpack("<d", data)[0]
    if t in ("float",):
        data = pread_mem(pid, addr, 4)
        if len(data) != 4:
            raise IOError("short read on float")
        return struct.unpack("<f", data)[0]
    if t in ("int",):
        data = pread_mem(pid, addr, 4)
        if len(data) != 4:
            raise IOError("short read on int")
        return struct.unpack("<i", data)[0]
    # default: read pointer-sized integer
    ptr64 = is_64_bit(pid)
    n = 8 if ptr64 else 4
    data = pread_mem(pid, addr, n)
    if len(data) != n:
        raise IOError("short read on value")
    return struct.unpack("<Q" if ptr64 else "<I", data)[0]


def find_variable_value(
    process_name: str,
    base_off: int,
    offsets: list[int],
    module_name: str = "",
    var_type: str = "string256",
    pid: int = None,
):
    if not pid:
        pid = (
            find_wine_process(process_name)[0]["pid"]
            if find_wine_process(process_name)
            else None
        )
    if not pid:
        return None

    try:

        # 2) Resolve base address
        # Default to main module '<process_name>.exe' if not provided
        module_name = module_name or (process_name + ".exe")
        mod_base = get_module_base(pid, module_name)
        if mod_base is None:
            raise RuntimeError(f"Module '{module_name}' not found in maps")
        base_addr = mod_base + base_off

        # 3) 32 vs 64 bit
        ptr64 = is_64_bit(pid)

        # 4) Resolve final address
        final_addr = follow_chain(pid, base_addr, offsets, ptr64=ptr64)

        # 5) Sanity check: mapped and readable?
        region = find_region(pid, final_addr, 1)
        if not region:
            raise RuntimeError(f"Address 0x{final_addr:x} is not mapped")
        start, end, perms, path = region
        if "r" not in perms:
            raise RuntimeError(
                f"Address 0x{final_addr:x} not readable (perms '{perms}')"
            )

        # 6) Read typed value
        return read_value(pid, final_addr, var_type)

    except OSError as e:
        # Helpful diagnostics for EIO/EPERM
        if e.errno == errno.EIO:
            print(
                "EIO while reading mem: usually unmapped address or not properly attached."
            )

        elif e.errno in (errno.EPERM, errno.EACCES):
            print("Permission denied: need root, CAP_SYS_PTRACE, or ptrace_scope=0.")
        return None
