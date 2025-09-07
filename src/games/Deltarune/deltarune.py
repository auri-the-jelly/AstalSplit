from __future__ import annotations

import os
import re
import struct
from typing import Optional

from aslrunner.proc_utils import (
    find_wine_process,
    iter_maps as _iter_maps,
    ptrace_attach,
    ptrace_detach,
    pread_mem,
    is_64_bit,
)


class DeltarunePlugin:
    """Game plugin for Deltarune.

    Responsibilities:
    - Signature scan to locate room array pointer and room ID pointer
    - Expose dynamic values like current.room and current.roomName
    """

    def __init__(self, process_name: str):
        self.process_name = process_name
        self.pid: Optional[int] = None
        self.ptr64: bool = True
        self._scanned: bool = False
        self._module_start: Optional[int] = None
        self._module_end: Optional[int] = None
        self._ptr_room_array: Optional[int] = None
        self._ptr_room_id: Optional[int] = None

    # -------- lifecycle --------
    def setup(self):
        # Lazy connect; real work happens on first update
        self._refresh_pid()

    def _refresh_pid(self):
        procs = find_wine_process(self.process_name)
        if procs:
            self.pid = procs[0]["pid"]
            self.ptr64 = is_64_bit(self.pid)
        else:
            self.pid = None

    # -------- low-level helpers --------
    def _iter_main_module(self):
        if not self.pid:
            return None
        # Choose the first .exe mapping with r-xp or r--p perms
        for start, end, perms, path in _iter_maps(self.pid):
            base = os.path.basename(path).lower()
            if base.endswith(".exe") and ("r-xp" in perms or "r--p" in perms):
                return (start, end, perms, path)
        return None

    def _read_mem(self, addr: int, size: int) -> bytes:
        if not self.pid:
            return b""
        return pread_mem(self.pid, addr, size)

    def _read_i32(self, addr: int) -> int:
        data = self._read_mem(addr, 4)
        if len(data) != 4:
            raise IOError("short read i32")
        return struct.unpack("<i", data)[0]

    def _read_ptr(self, addr: int) -> int:
        n = 8 if self.ptr64 else 4
        data = self._read_mem(addr, n)
        if len(data) != n:
            raise IOError("short read ptr")
        return struct.unpack("<Q" if self.ptr64 else "<I", data)[0]

    def _read_cstr(self, addr: int, max_len: int = 256) -> str:
        data = self._read_mem(addr, max_len)
        return data.split(b"\x00", 1)[0].decode("utf-8", "ignore")

    # -------- signature scan --------
    @staticmethod
    def _parse_pattern(pat: str):
        # "48 8B 05 ?? ?? ?? ??" -> list of (byte or None)
        out = []
        for tok in pat.split():
            if tok == "??":
                out.append(None)
            else:
                out.append(int(tok, 16))
        return out

    @staticmethod
    def _scan_first(hay: bytes, needle: list[Optional[int]]) -> int:
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

    def _ensure_scanned(self):
        if self._scanned:
            return
        self._refresh_pid()
        if not self.pid:
            return

        mod = self._iter_main_module()
        if not mod:
            return
        start, end, perms, path = mod
        size = end - start

        # Attach once for a stable read
        ptrace_attach(self.pid)
        try:
            data = self._read_mem(start, size)
            if not data:
                return

            # ptrRoomArray
            if self.ptr64:
                off = 5
                pat = self._parse_pattern(
                    "74 0C 48 8B 05 ?? ?? ?? ?? 48 8B 04 D0"
                )
                idx = self._scan_first(data, pat)
                if idx >= 0:
                    addr = start + idx + off
                    disp = self._read_i32(addr)
                    self._ptr_room_array = addr + disp + 0x4
            else:
                off = 2
                pat = self._parse_pattern("8B 3D ?? ?? ?? ?? 2B EF")
                idx = self._scan_first(data, pat)
                if idx >= 0:
                    addr = start + idx + off
                    self._ptr_room_array = self._read_ptr(addr)

            # ptrRoomID
            if self.ptr64:
                off = 6
                pat = self._parse_pattern(
                    "48 ?? ?? ?? 3B 35 ?? ?? ?? ?? 41 ?? ?? ?? 49 ?? ?? E8 ?? ?? ?? ?? FF"
                )
                idx = self._scan_first(data, pat)
                if idx >= 0:
                    addr = start + idx + off
                    disp = self._read_i32(addr)
                    self._ptr_room_id = addr + disp + 0x4
            else:
                off = 2
                pat = self._parse_pattern(
                    "FF 35 ?? ?? ?? ?? E8 ?? ?? ?? ?? 83 C4 04 50 68"
                )
                idx = self._scan_first(data, pat)
                if idx >= 0:
                    addr = start + idx + off
                    self._ptr_room_id = self._read_ptr(addr)

            self._module_start = start
            self._module_end = end
            self._scanned = True
        finally:
            ptrace_detach(self.pid)

    # -------- per-tick update --------
    def update(self, interpreter):
        # Ensure we have a pid and signatures once process is available
        if not self.pid:
            self._refresh_pid()
        if not self.pid:
            return
        if not self._scanned:
            self._ensure_scanned()
        if not (self._ptr_room_array and self._ptr_room_id):
            return

        # Read current room and name
        ptrace_attach(self.pid)
        try:
            room_id = self._read_i32(self._ptr_room_id)
            arr_main = self._read_ptr(self._ptr_room_array)
            if arr_main == 0:
                interpreter.current["current.room"] = room_id
                interpreter.current["current.roomName"] = ""
                return

            stride = 8 if self.ptr64 else 4
            item_ptr_addr = arr_main + (room_id * stride)
            item_ptr = self._read_ptr(item_ptr_addr)
            if item_ptr == 0:
                room_name = ""
            else:
                room_name = self._read_cstr(item_ptr, 64)

            interpreter.current["current.room"] = room_id
            interpreter.current["current.roomName"] = room_name
        finally:
            ptrace_detach(self.pid)
