#!/usr/bin/env python3
import os, re, time, struct, ctypes, ctypes.util, errno
from dataclasses import dataclass



@dataclass(frozen=True)
class _MapEntry:
    start: int
    end: int
    perms: str
    path: str
    base: str


@dataclass
class _MapsCacheEntry:
    timestamp: float
    start_time: float
    maps: list


_PID_CACHE: dict[str, tuple[int, float]] = {}
_PID_START_TIMES: dict[int, float] = {}
_MAPS_CACHE: dict[int, _MapsCacheEntry] = {}
_MODULE_BASE_CACHE: dict[tuple[int, str], tuple[float, int]] = {}
_PTR_SIZE_CACHE: dict[int, tuple[float, bool]] = {}

_MAPS_CACHE_TTL = 0.5  # seconds between /proc/<pid>/maps refreshes


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


# -------- caching helpers --------
def _get_proc_start_time(pid: int):
    try:
        return os.stat(f"/proc/{pid}").st_ctime
    except FileNotFoundError:
        return None
    except PermissionError:
        return None


def _purge_module_cache_for_pid(pid: int):
    for key in [key for key in _MODULE_BASE_CACHE if key[0] == pid]:
        _MODULE_BASE_CACHE.pop(key, None)


def _invalidate_pid(pid: int):
    _MAPS_CACHE.pop(pid, None)
    _PTR_SIZE_CACHE.pop(pid, None)
    _PID_START_TIMES.pop(pid, None)
    _purge_module_cache_for_pid(pid)
    for name, (cached_pid, _) in list(_PID_CACHE.items()):
        if cached_pid == pid:
            _PID_CACHE.pop(name, None)


def _register_pid(process_name: str, pid: int):
    start_time = _get_proc_start_time(pid)
    if start_time is None:
        return None
    _PID_CACHE[process_name] = (pid, start_time)
    _PID_START_TIMES[pid] = start_time
    return pid


def _ensure_pid_registered(pid: int):
    if pid is None:
        return False
    if pid in _PID_START_TIMES:
        current = _get_proc_start_time(pid)
        if current is None:
            _invalidate_pid(pid)
            return False
        if current != _PID_START_TIMES[pid]:
            _invalidate_pid(pid)
            return False
        return True
    start_time = _get_proc_start_time(pid)
    if start_time is None:
        return False
    _PID_START_TIMES[pid] = start_time
    return True


def _get_cached_pid(process_name: str):
    cached = _PID_CACHE.get(process_name)
    if cached:
        pid, start_time = cached
        current = _get_proc_start_time(pid)
        if current is not None and current == start_time:
            _PID_START_TIMES[pid] = start_time
            return pid
        _invalidate_pid(pid)
    matches = find_wine_process(process_name)
    if not matches:
        return None
    pid = matches[0]["pid"]
    return _register_pid(process_name, pid)


def _read_maps(pid: int):
    entries = []
    path = f"/proc/{pid}/maps"
    with open(path, "r") as f:
        for line in f:
            rng, perms, offset, dev, inode, *raw_path = line.strip().split()
            start_s, end_s = rng.split("-")
            start = int(start_s, 16)
            end = int(end_s, 16)
            mapped_path = raw_path[0] if raw_path else ""
            base = os.path.basename(mapped_path).lower() if mapped_path else ""
            entries.append(_MapEntry(start, end, perms, mapped_path, base))
    return entries


def _get_maps(pid: int, use_cache: bool = True):
    if not _ensure_pid_registered(pid):
        raise FileNotFoundError
    start_time = _PID_START_TIMES.get(pid)
    if use_cache:
        cache = _MAPS_CACHE.get(pid)
        now = time.monotonic()
        if cache and cache.start_time == start_time and (now - cache.timestamp) <= _MAPS_CACHE_TTL:
            return cache.maps
    try:
        maps = _read_maps(pid)
    except FileNotFoundError:
        _invalidate_pid(pid)
        raise
    now = time.monotonic()
    _purge_module_cache_for_pid(pid)
    _MAPS_CACHE[pid] = _MapsCacheEntry(timestamp=now, start_time=start_time or 0.0, maps=maps)
    return maps


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
def iter_maps(pid, use_cache: bool = True):
    for entry in _get_maps(pid, use_cache=use_cache):
        yield (entry.start, entry.end, entry.perms, entry.path)


def find_region(pid, addr, size=1):
    for start, end, perms, path in iter_maps(pid):
        if start <= addr and addr + size <= end:
            return (start, end, perms, path)
    return None


def get_module_base(pid, needle):
    if os.path.exists(needle):
        needle = os.path.basename(needle)
    needle = (needle or "").lower()
    if not needle:
        return None
    start_time = _PID_START_TIMES.get(pid)
    cache_key = (pid, needle)
    if start_time is not None:
        cached = _MODULE_BASE_CACHE.get(cache_key)
        if cached and cached[0] == start_time:
            return cached[1]
    for entry in _get_maps(pid):
        if (
            needle in entry.base
            and ("r--p" in entry.perms or "r-xp" in entry.perms or "rw-p" in entry.perms)
        ):
            if start_time is not None:
                _MODULE_BASE_CACHE[cache_key] = (start_time, entry.start)
            return entry.start
    return None


def get_all_modules(pid):
    modules = []
    for entry in _get_maps(pid):
        if ".exe" in entry.base:
            modules.append(
                {
                    "start": entry.start,
                    "end": entry.end,
                    "perms": entry.perms,
                    "path": entry.path,
                }
            )
    return modules


def get_module_memory(pid, needle):
    if os.path.exists(needle):
        needle = os.path.basename(needle)
    needle = needle.lower()
    for entry in _get_maps(pid):
        if needle == entry.base and ("r--p" in entry.perms or "r-xp" in entry.perms):
            size = entry.end - entry.start
            try:
                return _process_vm_read(pid, entry.start, size)
            except OSError:
                # Fall back to empty on failure; callers handle None/empty
                return None
    return None


# -------- safe remote read (process_vm_readv) --------
def pread_mem(pid: int, addr: int, size: int) -> bytes:
    return _process_vm_read(pid, addr, size)


# -------- pointer chain --------
def is_64_bit(pid):
    if not _ensure_pid_registered(pid):
        return struct.calcsize("P") == 8
    start_time = _PID_START_TIMES.get(pid)
    cached = _PTR_SIZE_CACHE.get(pid)
    if cached and start_time is not None and cached[0] == start_time:
        return cached[1]

    # Prefer the executable symlink for architecture detection
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
        with open(exe, "rb") as f:
            hdr = f.read(5)
            if hdr[:4] == b"\x7fELF":
                result = hdr[4] == 2  # 2=ELFCLASS64, 1=ELFCLASS32
                if start_time is not None:
                    _PTR_SIZE_CACHE[pid] = (start_time, result)
                return result
    except Exception:
        pass

    # Fallback: scan mapped executable regions
    for _, _, perms, path in iter_maps(pid):
        if "r-xp" in perms and path:
            try:
                with open(path, "rb") as f:
                    hdr = f.read(5)
                    if hdr[:4] == b"\x7fELF":
                        result = hdr[4] == 2
                        if start_time is not None:
                            _PTR_SIZE_CACHE[pid] = (start_time, result)
                        return result
            except Exception:
                continue

    # Last resort: assume current Python's pointer size
    result = struct.calcsize("P") == 8
    if start_time is not None:
        _PTR_SIZE_CACHE[pid] = (start_time, result)
    return result


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
    if pid is None:
        pid = _get_cached_pid(process_name)
    else:
        if not _ensure_pid_registered(pid):
            pid = _get_cached_pid(process_name)
    if pid is None:
        return None

    try:
        if not _ensure_pid_registered(pid):
            return None

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
