from __future__ import annotations

import os
import re
import struct
import hashlib
from typing import Optional
from pathlib import Path

from aslrunner.proc_utils import (
    find_wine_process,
    iter_maps as _iter_maps,
    ptrace_attach,
    ptrace_detach,
    pread_mem,
    is_64_bit,
    get_all_modules,
)


class SigScanner:
    def __init__(self, process, base_addr, mem_size):
        self.process = process
        self.base_addr = base_addr
        self.mem_size = mem_size
        self.memory = None
        if self.process:
            self.memory = get_module_memory(self.process["pid"], self.process["path"])

        def Scan(self):
            pass


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
        # Store pointers in a dictionary: name -> address of the pointer/value slot
        # Keys used: 'array' (room name array base pointer slot), 'roomid' (current room id slot)
        self._ptrs: dict[str, Optional[int]] = {"array": None, "roomid": None}
        # Signature metadata prepared from ASL init: name -> { 'x64': {off, sig}, 'x86': {off, sig} }
        self._asl_sigs: Optional[dict] = None

    # -------- lifecycle --------
    def setup(self):
        # Lazy connect; real work happens on first update
        self._refresh_pid()
        # Prepare ASL-derived signatures early
        try:
            self.initialize(None)
        except Exception:
            pass

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
        # Ensure ASL signatures are loaded
        if self._asl_sigs is None:
            try:
                self.initialize(None)
            except Exception:
                return

        # Attach once for a stable read
        ptrace_attach(self.pid)
        try:
            data = self._read_mem(start, size)
            if not data:
                return

            asl_sigs = self._asl_sigs

            # ptrRoomArray -> store address of the slot that holds the pointer to the array
            if self.ptr64:
                if asl_sigs is None or "ptrRoomArray" not in asl_sigs:
                    return  # require ASL-provided signatures; no hardcoded fallbacks
                off = asl_sigs["ptrRoomArray"]["x64"]["off"]
                pat = self._parse_pattern(asl_sigs["ptrRoomArray"]["x64"]["sig"])
                idx = self._scan_first(data, pat)
                if idx >= 0:
                    addr = start + idx + off
                    disp = self._read_i32(addr)
                    self._ptrs["array"] = addr + disp + 0x4
            else:
                if asl_sigs is None or "ptrRoomArray" not in asl_sigs:
                    return
                off = asl_sigs["ptrRoomArray"]["x86"]["off"]
                pat = self._parse_pattern(asl_sigs["ptrRoomArray"]["x86"]["sig"])
                idx = self._scan_first(data, pat)
                if idx >= 0:
                    addr = start + idx + off
                    self._ptrs["array"] = self._read_ptr(addr)

            # ptrRoomID -> store address of the slot that holds the current room id
            if self.ptr64:
                if asl_sigs is None or "ptrRoomID" not in asl_sigs:
                    return
                off = asl_sigs["ptrRoomID"]["x64"]["off"]
                pat = self._parse_pattern(asl_sigs["ptrRoomID"]["x64"]["sig"])
                idx = self._scan_first(data, pat)
                if idx >= 0:
                    addr = start + idx + off
                    disp = self._read_i32(addr)
                    self._ptrs["roomid"] = addr + disp + 0x4
            else:
                if asl_sigs is None or "ptrRoomID" not in asl_sigs:
                    return
                off = asl_sigs["ptrRoomID"]["x86"]["off"]
                pat = self._parse_pattern(asl_sigs["ptrRoomID"]["x86"]["sig"])
                idx = self._scan_first(data, pat)
                if idx >= 0:
                    addr = start + idx + off
                    self._ptrs["roomid"] = self._read_ptr(addr)

            self._module_start = start
            self._module_end = end
            self._scanned = True
        finally:
            ptrace_detach(self.pid)

    def initialize(self, interpreter):
        # Parse ASL init block for signature patterns and offsets
        try:
            here = os.path.dirname(__file__)
            asl_path = os.path.normpath(
                os.path.join(here, "..", "..", "aslrunner", "Deltarune.asl")
            )
            with open(asl_path, "r", encoding="utf-8") as f:
                txt = f.read()
        except Exception:
            self._asl_sigs = None
            return

        pattern = re.compile(
            r"""
            (?ms)
            (?:^|;)\s*
            (?P<lhs>(?:IntPtr|var)\s+)?(?P<name>(?:vars\.)?[A-Za-z_][A-Za-z0-9_]*)\s*=\s*vars\.x64\s*\?\s*
            (?P<fn64>[A-Za-z_][A-Za-z0-9_\.]*)\s*\(\s*(?P<off64>\d+)\s*,\s*\"(?P<sig64>[^\"]+)\"\s*\)\s*:\s*
            (?P<fn32>[A-Za-z_][A-Za-z0-9_\.]*)\s*\(\s*(?P<off32>\d+)\s*,\s*\"(?P<sig32>[^\"]+)\"\s*\)\s*;
            """,
            re.M | re.S,
        )

        results: dict[str, dict] = {}
        for m in pattern.finditer(txt):
            raw_name = m.group("name")
            name = raw_name.split(".", 1)[-1]  # strip optional 'vars.'
            try:
                results[name] = {
                    "x64": {"off": int(m.group("off64")), "sig": m.group("sig64")},
                    "x86": {"off": int(m.group("off32")), "sig": m.group("sig32")},
                }
            except Exception:
                continue

        self._asl_sigs = results or None

        game_hash = ""
        try:
            exe_path = get_all_modules(self.pid)[0]["path"]
            data_win = Path(exe_path).parent / "data.win"
            with open(data_win, "rb") as f:
                data = f.read()
                game_hash = hashlib.md5(data).hexdigest().upper()
        except Exception:
            game_hash = ""
        return game_hash

    # -------- per-tick update --------
    def update(self, interpreter):
        # Ensure we have a pid and signatures once process is available
        if not self.pid:
            self._refresh_pid()
        if not self.pid:
            return
        if not self._scanned:
            self._ensure_scanned()
        if not (self._ptrs.get("array") and self._ptrs.get("roomid")):
            return

        # Read current room and name
        ptrace_attach(self.pid)
        try:
            room_id = self._read_i32(self._ptrs["roomid"])
            arr_main = self._read_ptr(self._ptrs["array"])
            if arr_main == 0:
                interpreter.current["room"] = room_id
                interpreter.current["roomName"] = ""
                return

            stride = 8 if self.ptr64 else 4
            item_ptr_addr = arr_main + (room_id * stride)
            item_ptr = self._read_ptr(item_ptr_addr)
            if item_ptr == 0:
                room_name = ""
            else:
                room_name = self._read_cstr(item_ptr, 64)

            interpreter.current["room"] = room_id
            interpreter.current["roomName"] = room_name
        finally:
            ptrace_detach(self.pid)
