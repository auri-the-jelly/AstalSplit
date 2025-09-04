# region Imports
import math
import time
import re
import ast
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from gi.repository import GObject, GLib, Gio, Gtk
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


class ASLSettings(Gio.ListStore):
    __gtype_name__ = "ASLSettings"

    def __init__(self):
        super().__init__(item_type=GObject.TYPE_PYOBJECT)
        self.settings = {}

    def add(self, setting_name, setting_value, display_name, parent=""):
        self.settings[setting_name] = {
            "value": setting_value,
            "display_name": display_name,
            "parent": parent or None,
        }

    def set_tooltip(self, setting_name, tooltip):
        if setting_name in self.settings:
            self.settings[setting_name]["tooltip"] = tooltip


class ASLInterpreter(GObject.Object):
    __gtype__name__ = "ASLInterpreter"

    def __init__(self, asl_script: str):
        super().__init__()
        self.asl_script = open(asl_script, "r", encoding="utf-8").read().splitlines()
        # region State Declarations
        self.states = self.find_states()
        # endregion
        # region Runtime State
        self.vars = {}
        self.current = {}
        self.old = {}
        self.settings = ASLSettings()
        self.brace_loc = 0
        # endregion
        # region Initialization
        self.initialize()
        # endregion

    # region Common Helpers
    class DotDict(dict):
        def __getattr__(self, key):
            return self.get(key)

        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def _translate_expr(self, expr: str) -> str:
        s = expr.strip().rstrip(",")
        # Convert C# verbatim strings @"..." to Python-safe quoted strings
        s = self._replace_verbatim_strings(s)
        # Protect '!=' during replacements
        s = s.replace("!=", " __NEQ__ ")
        s = s.replace("&&", " and ")
        s = s.replace("||", " or ")
        # logical not: bare '!'
        s = re.sub(r"(?<![=!])!\s*", " not ", s)
        s = s.replace(" __NEQ__ ", " != ")
        # Methods
        s = re.sub(r"\.EndsWith\s*\(", ".endswith(", s)
        # obj.Contains(arg) -> (arg in obj)
        s = re.sub(r"([A-Za-z0-9_\.]+)\.Contains\s*\(([^()]+)\)", r"(\2 in \1)", s)
        # C# null
        s = re.sub(r"==\s*null", " is None", s, flags=re.IGNORECASE)
        s = re.sub(r"!=\s*null", " is not None", s, flags=re.IGNORECASE)
        # Already normalized verbatim strings above
        # Ternary cond ? a : b (single-level)
        if "?" in s and ":" in s:
            s = self._convert_ternary(s)
        return s

    def _replace_verbatim_strings(self, s: str) -> str:
        # Replace C# verbatim strings @"..." with Python-safe quoted strings
        # Handles doubled quotes inside: "" -> "
        def repl(m: re.Match) -> str:
            jls_extract_var = 1
            content = m[jls_extract_var]
            content = content.replace('""', '"')
            # In C# verbatim, backslashes are literal; escape them for Python
            content = content.replace("\\", "\\\\")
            return f'"{content}"'

        # Match @"..." allowing doubled quotes inside
        pattern = re.compile(r"@\"((?:[^\"]|\"\")*)\"")
        return pattern.sub(repl, s)

    # endregion

    def _convert_ternary(self, s: str) -> str:
        # Robust converter for patterns like (cond ? (a) : (b)) within a string
        i = 0
        while True:
            q = s.find("?", i)
            if q == -1:
                break
            # find last '(' before '?'
            l = s.rfind("(", 0, q)
            if l == -1 or ")" in s[l:q]:
                i = q + 1
                continue
            # parse (a)
            j = q + 1
            while j < len(s) and s[j].isspace():
                j += 1
            if j >= len(s) or s[j] != "(":
                i = q + 1
                continue
            depth = 0
            k = j
            while k < len(s):
                if s[k] == "(":
                    depth += 1
                elif s[k] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if k >= len(s):
                break
            a = s[j + 1 : k]
            # find ':' then (b)
            m = k + 1
            while m < len(s) and s[m].isspace():
                m += 1
            if m >= len(s) or s[m] != ":":
                i = q + 1
                continue
            m += 1
            while m < len(s) and s[m].isspace():
                m += 1
            if m >= len(s) or s[m] != "(":
                i = q + 1
                continue
            depth = 0
            n = m
            while n < len(s):
                if s[n] == "(":
                    depth += 1
                elif s[n] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                n += 1
            if n >= len(s):
                break
            b = s[m + 1 : n]
            cond = s[l + 1 : q].strip()
            replaced = f"{a.strip()} if ({cond}) else {b.strip()}"
            end_idx = n + 1
            if end_idx < len(s) and s[end_idx] == ")":
                end_idx += 1
            s = s[:l] + replaced + s[end_idx:]
            i = l + len(replaced)
        return s

    def _make_pyfunc(self, params: list[str], expr: str):
        py_expr = self._translate_expr(expr)
        code = compile(py_expr, "<asl-func>", "eval")

        def fn(*args):
            local_env = {name: arg for name, arg in zip(params, args)}
            return eval(code, {"__builtins__": {}}, local_env)

        return fn

    def _parse_dictionary_of_funcs(self) -> dict:
        d: dict[str, object] = {}
        n = len(self.asl_script)
        # Move to opening '{'
        while self.i < n and "{" not in self.asl_script[self.i]:
            self.i += 1
        # Inside block
        self.i += 1
        while self.i < n:
            line = self.asl_script[self.i].strip()
            if line.startswith("}"):  # end of dict
                break
            # {"Key", (a,b,c) => expr},
            m = re.match(r'^\{\s*"([^"]+)"\s*,\s*\(([^)]*)\)\s*=>\s*(.*)\}\s*,?$', line)
            if m:
                key = m.group(1)
                params = [p.strip() for p in m.group(2).split(",") if p.strip()]
                expr = m.group(3)
                d[key] = self._make_pyfunc(params, expr)
            self.i += 1
        return d

    def _parse_switch_func(self):
        # Parse a Func<...> lambda that contains a switch(ver) with return statements
        header = self.asl_script[self.i]
        m = re.search(r"\(\(([^)]*)\)\s*=>", header)
        if not m and self.i + 1 < len(self.asl_script):
            # Sometimes params spill to next line
            header = self.asl_script[self.i + 1]
            m = re.search(r"\(\(([^)]*)\)\s*=>", header)
            if m:
                self.i += 1
        if not m:
            return None
        params = [p.strip() for p in m.group(1).split(",") if p.strip()]

        # Advance to the opening '{'
        n = len(self.asl_script)
        while self.i < n and "{" not in self.asl_script[self.i]:
            self.i += 1
        # Step into block
        self.i += 1

        case_exprs: dict[str, callable] = {}
        default_fn = None
        current_case = None

        while self.i < n:
            line = self.asl_script[self.i].strip()
            if line.startswith("});"):
                break
            if line.startswith("}"):
                current_case = None
                self.i += 1
                continue
            mc = re.match(r'^case\s+"([^"]+)"\s*:\s*$', line)
            if mc:
                current_case = mc.group(1)
                self.i += 1
                continue
            if re.match(r"^default\s*:\s*$", line):
                current_case = "__DEFAULT__"
                self.i += 1
                continue
            mr = re.match(r"^return\s+(.*);\s*$", line)
            if mr and current_case is not None:
                expr = mr.group(1).strip()
                fn = self._make_pyfunc(params, expr)
                if current_case == "__DEFAULT__":
                    default_fn = fn
                else:
                    case_exprs[current_case] = fn
                self.i += 1
                continue
            self.i += 1

        def _fn(*args):
            local = {name: arg for name, arg in zip(params, args)}
            ver = local.get("ver")
            if isinstance(ver, str) and ver in case_exprs:
                return case_exprs[ver](*args)
            if default_fn is not None:
                return default_fn(*args)
            return False

        return _fn

    # region Initialization Parsing
    def initialize(self):
        self.vars = {}
        self.i = 0
        init_line = 0
        n = len(self.asl_script)
        current_parent = ""
        while self.i < n:
            if self.asl_script[self.i].strip().startswith("init"):
                init_line = self.i
                self.i += 1
                break
            if (
                self.asl_script[self.i].strip().startswith("vars.")
                and "=" in self.asl_script[self.i]
            ):
                var_name = (
                    self.asl_script[self.i].strip().strip(";").split("=")[0].strip()
                )
                var_value = self.identify_type(
                    self.asl_script[self.i].strip().strip(";").split("=")[1]
                )
                self.vars[var_name] = var_value
            self.i += 1
        # Parse multiline tooltips within the init block for UI usage
        if init_line:
            self._parse_init_tooltips(init_line)
        # Fallback/global parsing to ensure settings exist before tooltips
        self._parse_settings_and_tooltips_global()

    # endregion

    def identify_type(self, value: str):
        if value.startswith("//"):
            return "INVALID"
        value = value.strip().replace(",", "")
        n = len(self.asl_script)
        if "//" in value:
            value = value.split("//")[0]
        if ";" in value:
            value = value.strip().strip(";")
        if "new " in value:
            if value.startswith("new Dictionary"):
                return self._parse_dictionary_of_funcs()
            if value.startswith("new HashSet"):
                return set()
            return "INVALID"
        if "null" in value.strip():
            return None
        if value.strip() == "false":
            return False
        if value.strip() == "true":
            return True
        if "new[]" in value:
            self.i += 2
            array = []
            while self.i < n and self.asl_script[self.i].strip() != "};":
                array.append(
                    self.identify_type(self.asl_script[self.i].strip().strip(","))
                )
                self.i += 1
            return array
        # Capture (Action) lambdas into executable setters and method calls
        if "(Action)" in value:
            i = self.i + 2
            self.i += 2
            func = []
            while i < n and self.asl_script[i].strip() != "});":
                raw = self.asl_script[i]
                line = raw.strip()
                if line.startswith("//"):
                    i += 1
                    self.i += 1
                    continue
                # Assignment to vars.*
                if (
                    "=" in line
                    and line.endswith(";")
                    and line.split("=", 1)[0].strip().startswith("vars.")
                ):
                    name = line.split("=", 1)[0].strip()
                    rhs = line.split("=", 1)[1].strip().strip(";")
                    func.append(
                        lambda name=name, rhs=rhs: self.vars.__setitem__(
                            name, self.identify_type(rhs)
                        )
                    )
                # Method Clear(): vars.something.Clear();
                m_clear = re.match(
                    r"^([A-Za-z0-9_\.\[\]]+)\.Clear\s*\(\)\s*;\s*$", line
                )
                if m_clear:
                    target = m_clear.group(1)

                    def _do_clear(target=target):
                        obj = self.vars.get(target)
                        if hasattr(obj, "clear"):
                            obj.clear()
                        elif isinstance(obj, list):
                            obj[:] = []
                        elif isinstance(obj, dict):
                            obj.clear()
                        else:
                            self.vars[target] = None

                    func.append(_do_clear)
                # print("..."); simple string prints
                m_print = re.match(r"^print\s*\((.*)\)\s*;\s*$", line)
                if m_print:
                    arg = m_print.group(1).strip()
                    # Normalize verbatim strings and try to literal-eval
                    arg_norm = self._replace_verbatim_strings(arg)
                    msg_val = None
                    try:
                        if (arg_norm.startswith('"') and arg_norm.endswith('"')) or (
                            arg_norm.startswith("'") and arg_norm.endswith("'")
                        ):
                            msg_val = ast.literal_eval(arg_norm)
                    except Exception:
                        msg_val = None
                    if isinstance(msg_val, str):
                        func.append(lambda m=msg_val: print(m))
                # Future: Add/AddRange etc. as needed
                i += 1
                self.i += 1
            return func
        # Capture (Func<...>) lambdas with switch(ver) bodies (arrow may be on next line)
        if "(Func" in value:
            fn = self._parse_switch_func()
            if fn is not None:
                return fn
        # Single-line simple lambdas like (ver,org,cur) => condition
        if "=>" in value and value.startswith("("):
            m = re.match(r"^\(([^)]*)\)\s*=>\s*(.*)$", value)
            if m:
                params = [p.strip() for p in m.group(1).split(",") if p.strip()]
                expr = m.group(2)
                return self._make_pyfunc(params, expr)
            # If not matched, skip to the end of the block
            while self.i < n and self.asl_script[self.i].strip() != "});":
                self.i += 1
        if '"' in value:
            return value.strip().strip('"')
        if "." in value and value.strip().strip(".").isnumeric():
            try:
                return float(value)
            except ValueError:
                return "INVALID"
        if value.strip().isnumeric():
            try:
                return int(value)
            except ValueError:
                return "INVALID"
        return "INVALID"

    # region State Declarations Parsing
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

    # region Init Helpers: tooltips
    def _parse_init_tooltips(self, start_idx: int):
        """Scan the init { ... } block and collect settings.SetToolTip entries.
        Handles multi-line verbatim strings and concatenation with '+'.
        Stores results in self.settings['tooltips'][name] = tooltip.
        """
        i = start_idx
        n = len(self.asl_script)
        # find opening '{'
        while i < n and "{" not in self.asl_script[i]:
            i += 1
        if i >= n:
            return
        depth = self.asl_script[i].count("{") - self.asl_script[i].count("}")
        i += 1
        call_start_pat = re.compile(r"\bsettings\.SetToolTip\s*\(")
        while i < n and depth > 0:
            line = self.asl_script[i]
            depth += line.count("{") - line.count("}")
            # Capture setting declarations (single-line)
            if "settings.Add" in line:
                m = re.search(
                    r'settings\.Add\s*\(\s*"([^"]+)"\s*(?:,\s*(true|false))?\s*(?:,\s*"([^"]*)")?',
                    line,
                )
                if m:
                    name = m.group(1)
                    val = m.group(2)
                    desc = m.group(3) or ""
                    default = True if (val is None or val.lower() == "true") else False
                    try:
                        self.settings.add(name, default, desc)
                    except Exception:
                        pass
            if call_start_pat.search(line):
                call = line
                j = i + 1
                par = line.count("(") - line.count(")")
                while j < n:
                    call += "\n" + self.asl_script[j]
                    par += self.asl_script[j].count("(") - self.asl_script[j].count(")")
                    if par <= 0 and self.asl_script[j].strip().endswith(");"):
                        j += 1
                        break
                    j += 1
                # Extract name and tooltip expr
                m = re.search(
                    r'settings\.SetToolTip\s*\(\s*"([^"]+)"\s*,\s*(.*)\)\s*;',
                    call,
                    re.S,
                )
                if m:
                    name = m[1]
                    expr = m[2]
                    expr = self._replace_verbatim_strings(expr)
                    tooltip = self._eval_const_string_concat(expr)
                    if isinstance(tooltip, str):
                        self.settings.set_tooltip(name, tooltip)
                i = j
                continue
            i += 1

    def _eval_const_string_concat(self, expr: str):
        """Evaluate "literal" + "literal" + ... safely. Returns str or None."""
        try:
            tree = ast.parse(expr, mode="eval")
        except Exception:
            return None

        def walk(node):
            if isinstance(node, ast.Expression):
                return walk(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if hasattr(ast, "Str") and isinstance(node, ast.Str):
                return node.s
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                l = walk(node.left)
                r = walk(node.right)
                if isinstance(l, str) and isinstance(r, str):
                    return l + r
                return None
            return None

        return walk(tree)

    # endregion

    def _parse_settings_and_tooltips_global(self):
        """Global regex-based parsing for settings.Add and SetToolTip to ensure
        settings are captured even if init block parsing misses any."""
        text = "\n".join(self.asl_script)
        add_re = re.compile(
            r'settings\.Add\s*\(\s*"([^"]+)"\s*(?:,\s*(true|false))?\s*(?:,\s*"([^"]*)")?',
            re.I,
        )
        for m in add_re.finditer(text):
            name = m.group(1)
            val = (m.group(2) or "true").lower()
            desc = m.group(3) or ""
            default = True if val == "true" else False
            try:
                self.settings.add(name, default, desc)
            except Exception:
                pass

        tip_re = re.compile(
            r'settings\.SetToolTip\s*\(\s*"([^"]+)"\s*,\s*(.*?)\)\s*;', re.S
        )
        for m in tip_re.finditer(text):
            name = m.group(1)
            expr = self._replace_verbatim_strings(m.group(2))
            tooltip = self._eval_const_string_concat(expr)
            if isinstance(tooltip, str):
                try:
                    self.settings.set_tooltip(name, tooltip)
                except Exception:
                    pass

    # endregion


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
    for var_name, var_value in asl.vars.items():
        print(f"{var_name}: {var_value}")
        if type(var_value) == list:
            for f in var_value:
                if callable(f):
                    f()
