#!/usr/bin/env python3
"""
Standalone Deltarune room-name scanner.

This script:
 - Locates a running Wine process with a cmdline ending in 'DELTARUNE.exe'
 - Scans executable memory for signature(s) to find:
     • current room ID slot (x64/x86)
     • room-name array slot (x86 direct; x64 via generic MOV RIP+disp heuristic)
 - Reads and prints the current room name.

No imports from this repository are used; everything is self-contained.

Notes:
 - Requires Linux with process_vm_readv permissions (ptrace_scope=0 or CAP_SYS_PTRACE).
 - x64 room array scan uses a heuristic (generic '48 8B 05 ?? ?? ?? ??'), then validates
   candidates by checking that the indexed pointer decodes to a plausible room name.
"""

from __future__ import annotations

import os
import re
import sys
import time
import errno
import ctypes
import ctypes.util
import struct
from typing import Iterable, Optional, Tuple, List


# ---------- Process discovery ----------
def find_wine_process(
    name: str, *, cmd_regex: str | None = None, debug: bool = False
) -> list[dict]:
    """Find Wine processes whose cmdline contains a token ending in '{name}.exe'.

    Matching is flexible and supports:
    - Windows-style paths with backslashes and a drive letter (X:\\...\\name.exe)
    - Forward slashes (X:/.../name.exe)
    - Bare 'name.exe' anywhere in the combined cmdline
    """
    # Accept both backslash and forward slash; optional drive letter
    exe_pat_winpath = re.compile(
        rf"(?i)\b(?:[A-Za-z]:[\\/])?.*?[\\/]{re.escape(name)}\.exe\b"
    )
    exe_pat_simple = re.compile(rf"(?i)\b{re.escape(name)}\.exe\b")
    exe_pat_loose = re.compile(rf"(?i){re.escape(name)}\.exe")
    user_pat = re.compile(cmd_regex, re.I) if cmd_regex else None

    def list_pids():
        for entry in os.listdir("/proc"):
            if entry.isdigit():
                yield int(entry)

    def read_cmdline(pid: int) -> list[str]:
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
        # Check each arg and also the combined string for robustness
        joined = " ".join(argv)
        hit = None
        # User-provided regex has highest priority
        if user_pat and (
            user_pat.search(joined) or any(user_pat.search(a) for a in argv)
        ):
            hit = joined
        if not hit:
            for arg in argv:
                a = arg.strip('"')
                if (
                    exe_pat_winpath.search(a)
                    or exe_pat_simple.search(a)
                    or exe_pat_loose.search(a)
                ):
                    hit = arg
                    break
        if not hit and (
            exe_pat_winpath.search(joined)
            or exe_pat_simple.search(joined)
            or exe_pat_loose.search(joined)
        ):
            # Fallback: store the whole cmdline when only the combined match worked
            hit = joined
        if hit:
            matches.append({"pid": pid, "exe_arg": hit, "argv": argv})
        elif debug:
            # In debug, show processes that look like Wine-launched Windows apps
            if any(".exe" in a.lower() for a in argv):
                print(f"[dbg skip] pid={pid} argv={argv}")
    return matches


# ---------- libc: process_vm_readv ----------
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


class IOVec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]


if not hasattr(libc, "process_vm_readv"):
    print("process_vm_readv not available in libc; cannot proceed", file=sys.stderr)
    sys.exit(1)

libc.process_vm_readv.restype = ctypes.c_ssize_t
libc.process_vm_readv.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(IOVec),
    ctypes.c_ulong,
    ctypes.POINTER(IOVec),
    ctypes.c_ulong,
    ctypes.c_ulong,
]


def pread_mem(pid: int, addr: int, size: int) -> bytes:
    if size <= 0:
        return b""
    buf = ctypes.create_string_buffer(size)
    local = IOVec(ctypes.cast(buf, ctypes.c_void_p), size)
    remote = IOVec(ctypes.c_void_p(addr), size)
    nread = libc.process_vm_readv(
        pid, ctypes.byref(local), 1, ctypes.byref(remote), 1, 0
    )
    if nread < 0:
        e = ctypes.get_errno()
        raise OSError(e, f"process_vm_readv failed: {os.strerror(e)}")
    return buf.raw[:nread]


# ---------- /proc/pid helpers ----------
def iter_maps(pid: int) -> Iterable[Tuple[int, int, str, str]]:
    with open(f"/proc/{pid}/maps", "r") as f:
        for line in f:
            rng, perms, offset, dev, inode, *path = line.strip().split()
            start_s, end_s = rng.split("-")
            start = int(start_s, 16)
            end = int(end_s, 16)
            path = path[0] if path else ""
            yield (start, end, perms, path)


def is_64_bit(pid: int) -> bool:
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
        with open(exe, "rb") as f:
            hdr = f.read(5)
            if hdr[:4] == b"\x7fELF":
                return hdr[4] == 2
    except Exception:
        pass
    # Fallback: scan executable mappings
    for _, _, perms, path in iter_maps(pid):
        if "r-xp" in perms and path:
            try:
                with open(path, "rb") as f:
                    hdr = f.read(5)
                    if hdr[:4] == b"\x7fELF":
                        return hdr[4] == 2
            except Exception:
                continue
    return struct.calcsize("P") == 8


# ---------- Memory read helpers ----------
def read_i32(pid: int, addr: int) -> int:
    data = pread_mem(pid, addr, 4)
    if len(data) != 4:
        raise IOError("short read i32")
    return struct.unpack("<i", data)[0]


def read_u32(pid: int, addr: int) -> int:
    data = pread_mem(pid, addr, 4)
    if len(data) != 4:
        raise IOError("short read u32")
    return struct.unpack("<I", data)[0]


def read_ptr(pid: int, addr: int, ptr64: bool) -> int:
    n = 8 if ptr64 else 4
    data = pread_mem(pid, addr, n)
    if len(data) != n:
        raise IOError("short read ptr")
    return struct.unpack("<Q" if ptr64 else "<I", data)[0]


def read_cstr(pid: int, addr: int, max_len: int = 256) -> str:
    try:
        data = pread_mem(pid, addr, max_len)
    except OSError:
        return ""
    return data.split(b"\x00", 1)[0].decode("utf-8", "ignore")


# ---------- Signature scanning ----------
def parse_pattern(pat: str) -> List[Optional[int]]:
    out: list[Optional[int]] = []
    for tok in pat.split():
        if tok == "??":
            out.append(None)
        else:
            out.append(int(tok, 16))
    return out


def scan_first(hay: bytes, needle: List[Optional[int]]) -> int:
    n = len(needle)
    limit = len(hay) - n + 1
    for i in range(limit):
        ok = True
        for j, b in enumerate(needle):
            if b is None:
                continue
            if hay[i + j] != b:
                ok = False
                break
        if ok:
            return i
    return -1


def scan_region(pid: int, start: int, end: int, sig: str, offset: int) -> int:
    """Return absolute address of match+offset within [start,end), or -1."""
    needle = parse_pattern(sig)
    pat_len = len(needle)
    chunk = 1024 * 1024
    prev_tail = b""
    pos = start
    while pos < end:
        want = min(chunk, end - pos)
        try:
            cur = pread_mem(pid, pos, want)
        except OSError:
            prev_tail = b""
            pos += want
            continue
        if not cur:
            prev_tail = b""
            pos += want
            continue
        buf = prev_tail + cur
        idx = scan_first(buf, needle)
        if idx >= 0:
            abs_idx = (pos - len(prev_tail)) - start + idx
            return start + abs_idx + int(offset)
        if pat_len > 1:
            prev_tail = buf[-(pat_len - 1) :]
        else:
            prev_tail = b""
        pos += len(cur)
    return -1


# Known signatures (from ASL):
# x64: ptrRoomID
SIG_ROOMID_X64 = "48 ?? ?? ?? 3B 35 ?? ?? ?? ?? 41 ?? ?? ?? 49 ?? ?? E8 ?? ?? ?? ?? FF"
OFF_ROOMID_X64 = 6

# x86: ptrRoomID and ptrRoomArray
SIG_ROOMID_X86 = "FF 35 ?? ?? ?? ?? E8 ?? ?? ?? ?? 83 C4 04 50 68"
OFF_ROOMID_X86 = 2
SIG_ARRAY_X86 = "8B 3D ?? ?? ?? ?? 2B EF"
OFF_ARRAY_X86 = 2

# x64: ptrRoomArray (heuristic: generic MOV rax,[RIP+disp32])
SIG_ARRAY_X64_GENERIC = "01"
OFF_ARRAY_X64_GENERIC = 5


def find_roomid_slot(pid: int, ptr64: bool) -> Optional[int]:
    sig = SIG_ROOMID_X64 if ptr64 else SIG_ROOMID_X86
    off = OFF_ROOMID_X64 if ptr64 else OFF_ROOMID_X86
    # Prefer executable mappings (.exe/.dll), else any r-x anon
    regions = []
    for s, e, perms, path in iter_maps(pid):
        if "r" in perms:
            regions.append((s, e, perms, path))
    # Try named modules first
    for s, e, perms, path in regions:
        if not path:
            continue
        base = os.path.basename(path).lower()
        if not (base.endswith(".exe")):
            continue
        addr = scan_region(pid, s, e, sig, off)
        if addr >= 0:
            if ptr64:
                disp = read_i32(pid, addr)
                return addr + disp + 0x4
            else:
                return read_u32(pid, addr)
    # Fallback: anonymous exec regions
    for s, e, perms, path in regions:
        if path:
            continue
        addr = scan_region(pid, s, e, sig, off)
        if addr >= 0:
            if ptr64:
                disp = read_i32(pid, addr)
                return addr + disp + 0x4
            else:
                return read_u32(pid, addr)
    return None


def plausible_room_name(s: str) -> bool:
    if not s or len(s) < 4 or len(s) > 64:
        return False
    if not all(32 <= ord(c) < 127 for c in s):
        return False
    # Common prefixes in Deltarune: "room_", "PLACE_"
    if s.startswith("room_") or s.startswith("PLACE_"):
        return True
    # Allow a few other likely room-ish tokens
    return bool(re.match(r"^[A-Za-z0-9_./\\-]+$", s))


def find_array_slot_x64(pid: int, roomid_slot: int) -> Optional[int]:
    # Heuristic: search MOV rax, [RIP+disp32] only in the base module
    # and immediately following anonymous executable regions, then validate.
    needle = SIG_ARRAY_X64_GENERIC
    off = OFF_ARRAY_X64_GENERIC

    # Collect candidate regions: main .exe r-x segments and nearby anon r-x.
    exe_regions: list[tuple[int, int, str, str]] = []
    anon_following: list[tuple[int, int, str, str]] = []
    main_exe_end: int | None = None

    maps = list(iter_maps(pid))
    for s, e, perms, path in maps:
        base = os.path.basename(path).lower() if path else ""
        # Wine may append " (deleted)" or use memfd names; be flexible
        if path and (".exe" in base) and ("r" in perms):
            exe_regions.append((s, e, perms, path))
            # Track the furthest end across all main .exe segments
            if main_exe_end is None or e > main_exe_end:
                main_exe_end = e

    if main_exe_end is not None:
        # Include anonymous executable regions that start shortly after the main module
        for s, e, perms, path in maps:
            if path:
                continue
            if not ("r" in perms and "x" in perms):
                continue
            # Heuristic window: within 0x200000 bytes after main module end
            if s >= main_exe_end and (s - main_exe_end) <= 0x200000:
                anon_following.append((s, e, perms, path))

    regions_to_scan = exe_regions + anon_following

    candidates = []
    for s, e, perms, path in regions_to_scan:
        addr = scan_region(pid, s, e, needle, off)
        while addr >= 0:
            try:
                disp = read_i32(pid, addr)
                slot = addr + disp + 0x4
                candidates.append(slot)
            except OSError:
                pass
            # Continue scanning after this match
            next_from = (addr + 1) - off + s
            # Rescan remainder of this region by adjusting start
            addr = scan_region(pid, next_from, e, needle, off)

    # Validate each candidate by attempting to read a plausible room name
    for slot in candidates:
        try:
            room_id = read_i32(pid, roomid_slot)
            arr_main = read_ptr(pid, slot, True)
            if arr_main == 0 or room_id < 0 or room_id > 10000:
                continue
            item_ptr_addr = arr_main + (room_id * 8)
            item_ptr = read_ptr(pid, item_ptr_addr, True)
            if item_ptr == 0:
                continue
            name = read_cstr(pid, item_ptr, 64)
            if plausible_room_name(name):
                return slot
        except OSError:
            continue
    return None


def find_array_slot(pid: int, ptr64: bool, roomid_slot: int) -> Optional[int]:
    if ptr64:
        return find_array_slot_x64(pid, roomid_slot)
    # x86: use direct signature
    # Restrict search to main .exe r-x segments and nearby anon r-x segments
    addr = None
    maps = list(iter_maps(pid))
    exe_regions: list[tuple[int, int, str, str]] = []
    anon_following: list[tuple[int, int, str, str]] = []
    main_exe_end: int | None = None

    for s, e, perms, path in maps:
        base = os.path.basename(path).lower() if path else ""
        if path and (".exe" in base) and ("r" in perms):
            exe_regions.append((s, e, perms, path))
            if main_exe_end is None or e > main_exe_end:
                main_exe_end = e

    if main_exe_end is not None:
        for s, e, perms, path in maps:
            if path:
                continue
            if not ("r" in perms and "x" in perms):
                continue
            if s >= main_exe_end and (s - main_exe_end) <= 0x200000:
                anon_following.append((s, e, perms, path))

    for s, e, perms, path in exe_regions + anon_following:
        a = scan_region(pid, s, e, SIG_ARRAY_X86, OFF_ARRAY_X86)
        if a >= 0:
            addr = a
            break
    if addr is None:
        return None
    try:
        return read_u32(pid, addr)
    except OSError:
        return None


def main():
    # Args: [--name NAME] [--pid PID] [--cmd-regex REGEX] [--debug]
    target = "DELTARUNE"
    pid_override: int | None = None
    cmd_regex: str | None = None
    debug = False
    i = 1
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok in ("--name", "-n") and i + 1 < len(sys.argv):
            target = sys.argv[i + 1]
            i += 2
            continue
        if tok == "--pid" and i + 1 < len(sys.argv):
            try:
                pid_override = int(sys.argv[i + 1])
            except Exception:
                pass
            i += 2
            continue
        if tok == "--cmd-regex" and i + 1 < len(sys.argv):
            cmd_regex = sys.argv[i + 1]
            i += 2
            continue
        if tok == "--debug":
            debug = True
            i += 1
            continue
        i += 1

    if pid_override is not None:
        procs = [{"pid": pid_override, "exe_arg": "(pid override)", "argv": []}]
    else:
        procs = find_wine_process(target, cmd_regex=cmd_regex, debug=debug)
    if not procs:
        print(f"No Wine process with {target}.exe found.")
        sys.exit(1)
    pid = procs[0]["pid"]
    ptr64 = is_64_bit(pid)
    print(f"Found PID {pid} (ptr64={ptr64})")

    try:
        roomid_slot = find_roomid_slot(pid, ptr64)
        if not roomid_slot:
            print("Failed to locate room ID slot via signature scan.")
            sys.exit(2)
        print(f"roomid slot @ 0x{roomid_slot:X}")

        array_slot = find_array_slot(pid, ptr64, roomid_slot)
        if not array_slot:
            print("Failed to locate room array slot via signature scan.")
        else:
            print(f"array slot @ 0x{array_slot:X}")

        room_id = read_i32(pid, roomid_slot)
        if array_slot:
            arr_main = read_ptr(pid, array_slot, ptr64)
            if not arr_main:
                print("Room array base is NULL.")
        stride = 8 if ptr64 else 4
        print(f"Current room ID: {room_id}")
        item_ptr = read_ptr(pid, arr_main + room_id * stride, ptr64)
        name = read_cstr(pid, item_ptr, 64) if item_ptr else ""
        print(f"Room {room_id}: {name}")
    except OSError as e:
        if e.errno in (errno.EPERM, errno.EACCES):
            print("Permission denied. Try as root or adjust ptrace_scope.")
        else:
            print(f"Memory read error: {e}")
        sys.exit(10)


if __name__ == "__main__":
    main()
