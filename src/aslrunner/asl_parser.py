# region Imports
import math
import time
import re
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from gi.repository import GObject, GLib, Gio
from proc_utils import find_variable_value

# endregion


class ASLState(GObject.Object):
    __gtype__name__ = "ASLState"

    def __init__(
        self, process_name: str = "", version: str = "", variables: dict = None
    ):
        super().__init__()
        self.process_name = process_name
        self.version = version
        self.variables = variables if variables is not None else {}


class ASLInterpreter(GObject.Object):
    __gtype__name__ = "ASLInterpreter"

    def __init__(self, asl_script: str):
        super().__init__()
        self.asl_script = open(asl_script, "r", encoding="utf-8").read().splitlines()
        self.states = self.find_states()
        self.vars = []
        self.current = []
        self.old = []
        self.settings = []

    def find_states(self):
        """Parse the ASL script into ASLState objects with variables.

        Variables are parsed into a dict: name -> dict(type, base_module, offsets).
        - type: string like 'int', 'double', 'string128', ...
        - base_module: optional module string or None
        - offsets: list of integers (supports hex like 0x1234)
        """
        states: list[ASLState] = []

        def parse_state_header(line: str) -> tuple[str, str] | None:
            # state("PROC") or state("PROC", "VER")
            m = re.match(
                r"\s*state\(\s*\"([^\"]+)\"\s*(?:,\s*\"([^\"]+)\"\s*)?\)\s*", line
            )
            if not m:
                return None
            proc = m[1]
            ver = m[2] or ""
            return proc, ver

        def strip_inline_comments(s: str) -> str:
            # remove // comments and trim
            s = re.sub(r"/\*.*?\*/", "", s)
            s = re.sub(r"//.*", "", s)
            return s.strip()

        def parse_var_line(line: str, process_name: str):
            # Example:
            # double chapter : 0x6FE860, 0x30, 0x2F34, 0x80;
            # string128 text : "module.dll", 0x10, 0x20;
            s = strip_inline_comments(line)
            if not s or s == "}":
                return None
            if s.endswith(";"):
                s = s[:-1]
            if ":" not in s:
                return None
            left, right = s.split(":", 1)
            left = left.strip()
            right = right.strip()
            if " " not in left:
                return None
            var_type, var_name = left.split(None, 1)

            base_module = None
            parts = [p.strip() for p in right.split(",") if p.strip()]
            idx = 0
            if parts and parts[0].startswith('"') and parts[0].endswith('"'):
                base_module = parts[0][1:-1]
                idx = 1
            if base_module is None:
                base_module = f"{process_name}.exe"

            def to_int(tok: str) -> int:
                tok = tok.lower()
                return int(tok, 16) if tok.startswith("0x") else int(tok, 10)

            offsets_all = [to_int(p) for p in parts[idx:]]
            if not offsets_all:
                return None
            base_off = offsets_all[0]
            tail_offsets = offsets_all[1:]
            return var_name, {
                "type": var_type,
                "base_module": base_module,
                "base_off": base_off,
                "offsets": tail_offsets,
            }

        i = 0
        n = len(self.asl_script)
        while i < n:
            line = self.asl_script[i].strip()
            hdr = parse_state_header(line)
            if not hdr:
                i += 1
                continue
            process_name, version = hdr

            # Advance to block start '{'
            i += 1
            while i < n and "{" not in self.asl_script[i]:
                i += 1
            if i >= n:
                break

            # Inside block until matching '}' on a line
            i += 1
            vars_map: dict[str, dict] = {}
            while i < n:
                s = self.asl_script[i].strip()
                if s.startswith("}"):
                    break
                parsed = parse_var_line(self.asl_script[i], process_name)
                if parsed:
                    name, meta = parsed
                    vars_map[name] = meta
                i += 1

            states.append(
                ASLState(process_name=process_name, version=version, variables=vars_map)
            )

            # move past '}'
            i += 1

        return states


if __name__ == "__main__":
    asl = ASLInterpreter("src/aslrunner/Deltarune.asl")
    for state in asl.states:
        print(f"Process: {state.process_name}, Version: {state.version}")
        if state.version == "CH1-4 v1.04":
            for var_name, meta in state.variables.items():
                print(f"  {var_name}: {meta}")
                try:
                    val = find_variable_value(
                        state.process_name,
                        meta.get("base_off", 0),
                        meta.get("offsets", []),
                        meta.get("base_module", "DELTARUNE") or "",
                        meta.get("type", "string256"),
                    )
                    print(f"{var_name}:{val}")
                except Exception as e:
                    print(f"Error finding variable {var_name}: {e}")
