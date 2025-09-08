#!/usr/bin/env python3
import os, sys, time, struct, ctypes, ctypes.util, errno

# -------- ptrace helpers --------
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
PTRACE_ATTACH = 16
PTRACE_DETACH = 17


def ptrace_attach(pid):
    if libc.ptrace(PTRACE_ATTACH, pid, None, None) != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"ptrace(ATTACH) failed: {os.strerror(e)}")
    # wait for stop
    _, status = os.waitpid(pid, 0)
    if os.WIFSTOPPED(status) is False:
        raise RuntimeError("Target did not stop after attach")


def ptrace_detach(pid):
    if libc.ptrace(PTRACE_DETACH, pid, None, None) != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"ptrace(DETACH) failed: {os.strerror(e)}")


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
    for start, end, perms, path in iter_maps(pid):
        if needle and needle in path and "r--p" in perms:
            return start
    return None


# -------- safe /proc/pid/mem read --------
def pread_mem(pid, addr, size):
    fd = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
    try:
        return os.pread(fd, size, addr)
    finally:
        os.close(fd)


# -------- pointer chain --------
def is_64_bit(pid):
    # crude check: look at first executable mapping of main binary and infer ELF class
    for s, e, perms, path in iter_maps(pid):
        if (
            "r-xp" in perms
            and path
            and path.endswith((".exe", "(deleted)"))
            or path.endswith(".so")
            or path.startswith("/")
        ):
            try:
                with open(path, "rb") as f:
                    hdr = f.read(5)
                    # ELF: 0x7f 'E''L''F' then EI_CLASS at byte 4
                    if hdr[:4] == b"\x7fELF":
                        return hdr[4] == 2  # 2=ELFCLASS64, 1=ELFCLASS32
            except Exception:
                pass
    # fallback to arch of current Python (not perfect for Wine)
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
    for off in offsets[:-1]:
        ptr = read_ptr(pid, ptr + off, ptr64)
    return ptr + offsets[-1]


def read_cstring(pid, addr, maxlen=256):
    data = pread_mem(pid, addr, maxlen)
    return data.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")


# -------- main example --------
if __name__ == "__main__":
    # INPUTS YOU LIKELY NEED TO CHANGE:
    pid = 136224

    # Your pointer spec: string256 directory : 0x8B2818, 0x0;
    # If 0x8B2818 is RELATIVE to a module, set module_name.
    module_name = (
        "DELTARUNE.exe"  # e.g. "game.exe" for Wine targets; leave empty if absolute
    )
    base_off = 0x8B2818
    offsets = [0x0]

    try:
        # 1) Attach so /proc/<pid>/mem reads don’t EIO
        ptrace_attach(pid)

        # 2) Resolve base address
        if module_name:
            mod_base = get_module_base(pid, module_name)
            if mod_base is None:
                raise RuntimeError(f"Module '{module_name}' not found in maps")
            base_addr = mod_base + base_off
        else:
            base_addr = base_off  # treat as absolute

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

        # 6) Read the 256-byte string
        directory = read_cstring(pid, final_addr, 256)
        print("Directory:", directory)

    except OSError as e:
        # Helpful diagnostics for EIO/EPERM
        if e.errno == errno.EIO:
            print(
                "EIO while reading mem: usually unmapped address or not properly attached."
            )
        elif e.errno in (errno.EPERM, errno.EACCES):
            print("Permission denied: need root, CAP_SYS_PTRACE, or ptrace_scope=0.")
        raise
    finally:
        try:
            ptrace_detach(pid)
        except Exception:
            pass
