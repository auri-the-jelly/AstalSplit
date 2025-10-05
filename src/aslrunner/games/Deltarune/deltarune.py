from __future__ import annotations

import os
import re
import struct
import hashlib
import time
from typing import Optional
from pathlib import Path

# Prefer package-relative imports; fall back when run directly
try:
    from ...proc_utils import (
        find_wine_process,
        iter_maps as _iter_maps,
        pread_mem,
        is_64_bit,
        get_all_modules,
    )
except Exception:  # pragma: no cover
    from proc_utils import (
        find_wine_process,
        iter_maps as _iter_maps,
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
        self._main_module_base: Optional[int] = None
        # Store pointers in a dictionary: name -> address of the pointer/value slot
        # Keys used: 'array' (room name array base pointer slot), 'roomid' (current room id slot)
        self._ptrs: dict[str, Optional[int]] = {"array": None, "roomid": None}
        # Signature metadata prepared from ASL init: name -> { 'x64': {off, sig}, 'x86': {off, sig} }
        self._asl_sigs: Optional[dict] = None
        # Throttle expensive process/module discovery work
        self._last_pid_refresh: float = 0.0
        self._pid_rescan_interval: float = 0.5
        self._last_module_refresh: float = 0.0
        self._module_rescan_interval: float = 0.5
        self._cached_modules: Optional[list[dict]] = None
        self._cached_modules_ts: float = 0.0

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
        now = time.monotonic()

        # Fast-path: keep existing pid if process still alive
        if self.pid and os.path.exists(f"/proc/{self.pid}"):
            return

        if self.pid:
            # Process vanished; clear cached pointers so a new instance can be scanned
            self._reset_process_state()
            self.pid = None

        if now - self._last_pid_refresh < self._pid_rescan_interval:
            return
        self._last_pid_refresh = now

        procs = find_wine_process(self.process_name)
        if not procs:
            return

        new_pid = procs[0]["pid"]
        if new_pid != self.pid:
            self.pid = new_pid
            self.ptr64 = is_64_bit(self.pid)
            self._reset_process_state()
            self._ensure_main_module_cached(force=True)
        elif self._main_module_base is None:
            self._ensure_main_module_cached()

    def _reset_process_state(self):
        self._scanned = False
        self._module_start = None
        self._module_end = None
        self._main_module_base = None
        self._ptrs = {"array": None, "roomid": None}
        self._last_module_refresh = 0.0
        self._invalidate_module_cache()

    def _invalidate_module_cache(self):
        self._cached_modules = None
        self._cached_modules_ts = 0.0

    def _get_cached_modules(self, force: bool = False) -> list[dict]:
        if not self.pid:
            return []
        now = time.monotonic()
        if (
            not force
            and self._cached_modules is not None
            and now - self._cached_modules_ts < self._module_rescan_interval
        ):
            return self._cached_modules
        try:
            mods = get_all_modules(self.pid)
        except Exception:
            mods = []
        self._cached_modules = mods
        self._cached_modules_ts = now
        return mods

    def _ensure_main_module_cached(self, force: bool = False):
        if not self.pid:
            return
        now = time.monotonic()
        if (
            self._main_module_base is not None
            and not force
        ):
            return
        if not force and now - self._last_module_refresh < self._module_rescan_interval:
            return
        self._last_module_refresh = now
        try:
            mods = self._iter_main_module(force=force)
            if mods:
                self._main_module_base = mods[0][0]
        except Exception:
            self._main_module_base = None

    # -------- low-level helpers --------
    def _iter_main_module(self, force: bool = False):
        if not self.pid:
            return []
        # Prefer mapping that matches the process name (e.g., DELTARUNE.exe)
        preferred = f"{self.process_name.lower()}.exe"
        fallback = None
        pls_help = []
        modules = self._get_cached_modules(force=force)
        # If cache miss returned empty, try one forced refresh before bailing
        if not modules and not force:
            modules = self._get_cached_modules(force=True)
        for mod in modules:
            start = mod.get("start")
            end = mod.get("end")
            perms = mod.get("perms", "")
            path = mod.get("path", "")
            if start is None or end is None:
                continue
            base = os.path.basename(path).lower()
            if base == preferred and ("r" in perms):
                pls_help.append((start, end, perms, path))
            if fallback is None and base.endswith(".exe") and ("r" in perms):
                fallback = (start, end, perms, path)
        # If no exact match was found, fall back to the first executable mapping
        if not pls_help and fallback:
            return [fallback]
        return pls_help

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

    def _addr_with_main_base(self, addr: Optional[int]) -> Optional[int]:
        """Treat a small value as module-relative offset and add module base.

        Tries both the scanning module start (`_module_start`) and the main
        executable module base (DELTARUNE.exe), using the first applicable one.
        """
        if addr is None:
            return None
        # Ensure we have at least one candidate base
        if self._main_module_base is None and self._module_start is None:
            self._ensure_main_module_cached()

        candidates = []
        if self._module_start is not None:
            candidates.append(self._module_start)
        if (
            self._main_module_base is not None
            and self._main_module_base != self._module_start
        ):
            candidates.append(self._main_module_base)

        # Heuristic: if addr appears to be below any known module base, treat as offset
        for base in candidates:
            if addr < base:
                return base + addr
        return addr

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
        # Only consider scanning complete when both pointers are found.
        if self._scanned and self._ptrs.get("array") and self._ptrs.get("roomid"):
            return
        self._refresh_pid()
        if not self.pid:
            return

        def _try_scan_region(start, end, perms, path):
            size = end - start
            mod_name = os.path.basename(path) if path else "<anon>"
            print(
                f"[DeltarunePlugin] Scanning {mod_name} [{hex(start)}-{hex(end)}], size={size}"
            )
            # Ensure ASL signatures are loaded
            if self._asl_sigs is None:
                try:
                    self.initialize(None)
                except Exception:
                    return False

            # Robust, chunked reader to avoid EFAULT (bad address) on huge/guarded regions
            def _scan_pattern_in_region(sig: str, offset: int) -> int:
                if not sig:
                    return -1
                needle = self._parse_pattern(sig)
                pat_len = len(needle)
                chunk = 1024 * 1024  # 1 MiB
                prev_tail = b""
                pos = start
                while pos < end:
                    want = min(chunk, end - pos)
                    try:
                        cur = self._read_mem(pos, want)
                    except OSError:
                        # skip unreadable window; reset overlap
                        prev_tail = b""
                        pos += want
                        continue
                    if not cur:
                        prev_tail = b""
                        pos += want
                        continue
                    buf = prev_tail + cur
                    idx = self._scan_first(buf, needle)
                    if idx >= 0:
                        # absolute index relative to region start
                        abs_idx = (pos - len(prev_tail)) - start + idx
                        return abs_idx + int(offset)
                    # keep overlap to handle cross-boundary matches
                    if pat_len > 1:
                        prev_tail = buf[-(pat_len - 1) :]
                    else:
                        prev_tail = b""
                    pos += len(cur)
                return -1

            try:
                asl_sigs = self._asl_sigs

                # ptrRoomArray
                if asl_sigs is None or "ptrRoomArray" not in asl_sigs:
                    return False
                found_array = False
                off64 = asl_sigs["ptrRoomArray"].get("x64", {}).get("off")
                sig64 = asl_sigs["ptrRoomArray"].get("x64", {}).get("sig")
                if off64 is not None and sig64:
                    rel = _scan_pattern_in_region(sig64, off64)
                    if rel >= 0:
                        addr = start + rel
                        disp = self._read_i32(addr)
                        candidate = addr + disp + 0x4
                        # Do not overwrite if already found earlier (e.g., in main module)
                        if self._ptrs["array"] is None:
                            self._ptrs["array"] = candidate
                            self.ptr64 = True
                            print(
                                f"[DeltarunePlugin] ptrRoomArray slot @ 0x{self._ptrs['array']:X} (x64)"
                            )
                        else:
                            print(
                                f"[DeltarunePlugin] ptrRoomArray already set; keeping 0x{self._ptrs['array']:X} (found also in {mod_name})"
                            )
                        found_array = True
                if not found_array:
                    off32 = asl_sigs["ptrRoomArray"].get("x86", {}).get("off")
                    sig32 = asl_sigs["ptrRoomArray"].get("x86", {}).get("sig")
                    if off32 is not None and sig32:
                        rel = _scan_pattern_in_region(sig32, off32)
                        if rel >= 0:
                            addr = start + rel
                            raw = self._read_mem(addr, 4)
                            if len(raw) == 4:
                                candidate = struct.unpack("<I", raw)[0]
                                if self._ptrs["array"] is None:
                                    self._ptrs["array"] = candidate
                                    self.ptr64 = False
                                    print(
                                        f"[DeltarunePlugin] ptrRoomArray slot @ 0x{self._ptrs['array']:X} (x86)"
                                    )
                                else:
                                    print(
                                        f"[DeltarunePlugin] ptrRoomArray already set; keeping 0x{self._ptrs['array']:X} (found also in {mod_name})"
                                    )
                                found_array = True
                if not found_array:
                    print(
                        f"[DeltarunePlugin] ptrRoomArray pattern not found in module {mod_name}"
                    )

                # ptrRoomID
                if asl_sigs is None or "ptrRoomID" not in asl_sigs:
                    return False
                found_roomid = False
                off64 = asl_sigs["ptrRoomID"].get("x64", {}).get("off")
                sig64 = asl_sigs["ptrRoomID"].get("x64", {}).get("sig")
                if off64 is not None and sig64:
                    rel = _scan_pattern_in_region(sig64, off64)
                    if rel >= 0:
                        addr = start + rel
                        disp = self._read_i32(addr)
                        self._ptrs["roomid"] = addr + disp + 0x4
                        self.ptr64 = True
                        found_roomid = True
                        print(
                            f"[DeltarunePlugin] ptrRoomID slot @ 0x{self._ptrs['roomid']:X} (x64)"
                        )
                if not found_roomid:
                    off32 = asl_sigs["ptrRoomID"].get("x86", {}).get("off")
                    sig32 = asl_sigs["ptrRoomID"].get("x86", {}).get("sig")
                    if off32 is not None and sig32:
                        rel = _scan_pattern_in_region(sig32, off32)
                        if rel >= 0:
                            addr = start + rel
                            raw = self._read_mem(addr, 4)
                            if len(raw) == 4:
                                self._ptrs["roomid"] = struct.unpack("<I", raw)[0]
                                self.ptr64 = False
                                found_roomid = True
                                print(
                                    f"[DeltarunePlugin] ptrRoomID slot @ 0x{self._ptrs['roomid']:X} (x86)"
                                )
                if not found_roomid:
                    print(
                        f"[DeltarunePlugin] ptrRoomID pattern not found in module {mod_name}"
                    )

                self._module_start = start
                self._module_end = end
                if self._ptrs.get("array") and self._ptrs.get("roomid"):
                    self._scanned = True
                    return True
            except OSError as e:
                print(f"[DeltarunePlugin] Memory read error: {e}")
            return False

        # 1) Try the main executable mappings (preferred)
        for mod in self._iter_main_module():
            if not mod:
                continue
            # Scan the main executable mapping
            if _try_scan_region(*mod):
                return
            # Additionally, scan anonymous executable region(s) immediately after it
            try:
                _, mod_end, _, _ = mod
                # Look ahead for the first anonymous r-xp region that starts at or just after the main module
                for start, end, perms, path in _iter_maps(self.pid):
                    if path:
                        continue
                    if "r" not in perms or "x" not in perms:
                        continue
                    # Heuristic: within 0x200000 bytes after main module end
                    if start >= mod_end and (start - mod_end) <= 0x200000:
                        if _try_scan_region(start, end, perms, path):
                            return
                        # Only try the first such region
                        break
            except Exception:
                pass

        # 2) Fallback: scan all .exe/.dll modules until found
        try:
            for m in self._get_cached_modules():
                base = os.path.basename(m["path"]).lower()
                if not (base.endswith(".exe") or base.endswith(".dll")):
                    continue
                if "r" not in m["perms"]:
                    continue
                if _try_scan_region(m["start"], m["end"], m["perms"], m["path"]):
                    return
        except Exception:
            pass

        # 3) Last resort: scan anonymous executable mappings (no path)
        try:
            for start, end, perms, path in _iter_maps(self.pid):
                if path:
                    continue
                if "r" not in perms or "x" not in perms:
                    continue
                if _try_scan_region(start, end, perms, path):
                    return
        except Exception:
            pass

    def _convert_to_plaintext(self, text: str) -> str:
        """Convert Deltarune text with inline formatting codes to plain text.

        Example: "Hello {w}world{/w}!" -> "Hello world!"
        """
        # 1. Pause markers: ^<number>
        s = re.sub(r"\^\d+", "", s)

        # 2. Flow control
        s = re.sub(r"&", "", s)
        s = re.sub(r"/\%\%", "", s)
        s = re.sub(r"/\%", "", s)
        s = re.sub(r"/%", "", s)
        s = re.sub(r"%", "", s)
        s = re.sub(r"/\*", "", s)
        s = re.sub(r"\\C", "", s)

        # 3. Colors (\W, \X, \p, etc.)
        s = re.sub(r"\\[WXRROYGLBP]", "", s)  # single-letter codes
        s = re.sub(r"\\p", "", s)  # lowercase p

        # 4. Text modes (\T…)
        s = re.sub(r"\\T[a-zA-Z0-9]", "", s)

        # 5. Special sequences
        s = re.sub(r"#", "", s)  # octothorpe = newline marker in "free text"
        s = re.sub(r"\\z\d+", "", s)  # e.g. \z4

        # 6. Collapse extra spaces
        s = re.sub(r"\s+", " ", s).strip()

        return s

    def initialize(self, interpreter):
        # Parse ASL init block for signature patterns and offsets
        self._refresh_pid()
        try:
            here = os.path.dirname(__file__)
            if not interpreter:
                asl_path = os.path.normpath(
                    os.path.join(here, "..", "..", "..", "asl_scripts", "DELTARUNE.asl")
                )
            else:
                asl_path = interpreter.asl_path
            with open(asl_path, "r", encoding="utf-8") as f:
                txt = f.read()
        except Exception as e:
            print(f"[DeltarunePlugin] Failed to read ASL file: {e}")
            self._asl_sigs = None
            return

        # Robust, whitespace-insensitive capture of lines like:
        #   IntPtr ptrRoomArray = vars.x64 ? scan(5, "AA BB ?? CC") : scan(2, "...");
        #   vars.ptrRoomID = vars.x64 ? scan(6, "...") : scan(2, "...");
        pattern = re.compile(
            r"""
            (?:^|;)\s*                                            # start of statement
            (?: (?P<lhs>(?:IntPtr|var)) \s+ )?                    # optional type
            (?P<name>(?:vars\.)?[A-Za-z_][A-Za-z0-9_]*) \s* = \s* vars\.x64 \s* \? \s*
            (?P<fn64>[A-Za-z_][A-Za-z0-9_\.]*) \s* \( \s* (?P<off64>\d+) \s* , \s* " (?P<sig64>[^"]+) " \s* \) \s* : \s*
            (?P<fn32>[A-Za-z_][A-Za-z0-9_\.]*) \s* \( \s* (?P<off32>\d+) \s* , \s* " (?P<sig32>[^"]+) " \s* \) \s* ;
            """,
            re.M | re.S | re.X,
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
        # Helpful debug output so we know whether signatures were parsed
        if self._asl_sigs:
            parsed = ", ".join(sorted(self._asl_sigs.keys()))
            print(f"[DeltarunePlugin] Parsed ASL signatures: {parsed}")
        else:
            print("[DeltarunePlugin] No ASL signatures parsed")

        game_hash = ""
        try:
            modules = self._get_cached_modules(force=True)
            exe_path = modules[0]["path"]
            data_win = Path(exe_path).parent / "data.win"
            with open(data_win, "rb") as f:
                data = f.read()
                game_hash = hashlib.md5(data).hexdigest().upper()
        except Exception as e:
            game_hash = ""
        return game_hash

    # -------- per-tick update --------
    def update(self, interpreter):
        # Ensure we have a pid and signatures once process is available
        self._refresh_pid()
        if not self.pid:
            return None
        if not self._scanned:
            self._ensure_scanned()
        if not (self._ptrs.get("array") and self._ptrs.get("roomid")):
            # Fallback: if both are missing, try treating stored values as module-relative
            # offsets if present (user request: try reading pointer with offset at main module)
            pass

        try:
            # Primary attempt: use captured addresses as-is
            roomid_addr = self._ptrs.get("roomid")
            array_addr = self._ptrs.get("array")
            # array_addr = 170245D67
            # roomid_addr = 1408B27C8
            # Both beyond the confines of the memory map.
            if roomid_addr is None or array_addr is None:
                # If one is missing, try module-relative interpretation
                roomid_addr = self._addr_with_main_base(roomid_addr)
                array_addr = self._addr_with_main_base(array_addr)

            room_id = self._read_i32(roomid_addr)
            arr_main = self._read_ptr(array_addr)
            if arr_main == 0:
                interpreter.current["room"] = room_id
                interpreter.current["roomName"] = ""
                return [
                    "current.room = game.ReadValue<int>((IntPtr)vars.ptrRoomID);",
                    "current.roomName = vars.getRoomName();",
                ]

            stride = 8 if self.ptr64 else 4
            item_ptr_addr = arr_main + (room_id * stride)
            item_ptr = self._read_ptr(item_ptr_addr)
            if item_ptr == 0:
                room_name = ""
            else:
                room_name = self._read_cstr(item_ptr, 128)

            interpreter.current["room"] = room_id
            interpreter.current["roomName"] = room_name

            # Inform interpreter that these two lines are handled by the plugin
            return [
                "current.room = game.ReadValue<int>((IntPtr)vars.ptrRoomID);",
                "current.roomName = vars.getRoomName();",
            ]

        except Exception:
            # Secondary attempt: treat addresses as module-relative offsets and retry
            try:
                roomid_addr = self._addr_with_main_base(self._ptrs.get("roomid"))
                array_addr = self._addr_with_main_base(self._ptrs.get("array"))
                if roomid_addr is None or array_addr is None:
                    return None
                room_id = self._read_i32(roomid_addr)
                arr_main = self._read_ptr(array_addr)
                if arr_main == 0:
                    interpreter.current["room"] = room_id
                    interpreter.current["roomName"] = ""
                    return [
                        "current.room = game.ReadValue<int>((IntPtr)vars.ptrRoomID);",
                        "current.roomName = vars.getRoomName();",
                    ]

                stride = 8 if self.ptr64 else 4
                item_ptr_addr = arr_main + (room_id * stride)
                item_ptr = self._read_ptr(item_ptr_addr)
                room_name = "" if item_ptr == 0 else self._read_cstr(item_ptr, 64)

                interpreter.current["room"] = room_id
                interpreter.current["roomName"] = room_name
                return [
                    "current.room = game.ReadValue<int>((IntPtr)vars.ptrRoomID);",
                    "current.roomName = vars.getRoomName();",
                ]
            except Exception as e:
                return None
