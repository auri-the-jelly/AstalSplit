# region Imports
import math
import time
import re
import ast
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from gi.repository import GObject, GLib, Gio, Gtk
from proc_utils import (
    find_variable_value,
    find_wine_process,
    get_all_modules,
    is_64_bit,
    get_module_base,
    get_module_memory,
)
from game_plugins import load_plugin

# endregion


class ASLModule(GObject.Object):
    __gtype__name__ = "ASLModule"

    def __init__(self, process_name: str = None):
        super().__init__()
        if process_name and find_wine_process(process_name):
            self.process = find_wine_process(process_name)[0]
            self.modules = get_all_modules(self.process["pid"]) if self.process else []
            if self.modules:
                self.game = self.First()
                self.BaseAddress = get_module_base(
                    self.process["pid"], self.First()["path"]
                )
                self.ModuleMemorySize = get_module_memory(
                    self.process["pid"], self.First()["path"]
                )
                self.FileName = Path(self.First()["path"]).parent
            else:
                self.process = None

        else:
            self.process = None

    def First(self):
        return (
            get_all_modules(self.process["pid"])[0]
            if self.process and get_all_modules(self.process["pid"])
            else None
        )

    def is64Bit(self):
        return is_64_bit(self.process["pid"]) if self.process else False


class ASLState(GObject.Object):
    __gtype__name__ = "ASLState"

    def __init__(
        self, process_name: str = "", version: str = "", variables: dict = None
    ):
        super().__init__()
        self.process_name = process_name
        self.version = version
        self.variables = variables if variables is not None else {}


class ASLSetting(GObject.GObject):
    """
    A simple GObject so properties notify correctly and bind to widgets.
    """

    setting_name = GObject.Property(type=str)
    display_name = GObject.Property(type=str)
    setting_value = GObject.Property(type=bool, default=False)
    parent = GObject.Property(type=str, default="")

    def __init__(self, setting_name, display_name, setting_value=False, parent=""):
        super().__init__()
        self.setting_name = setting_name
        self.display_name = display_name
        self.setting_value = setting_value
        self.parent = parent


class ASLSettings:

    def __init__(self):
        super().__init__()
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

    def export_list_object(self):
        """Return a Gtk.TreeListModel for hierarchical settings.

        Items are ASLSetting GObjects with properties:
        - setting_name (id)
        - display_name
        - setting_value
        - parent (id of parent, empty string means top-level)
        """

        # Precreate GObjects so tree uses stable instances at all levels
        items: dict[str, ASLSetting] = {}
        for name, meta in self.settings.items():
            items[name] = ASLSetting(
                setting_name=name,
                display_name=meta.get("display_name", name),
                setting_value=meta.get("value", False),
                parent=meta.get("parent", "") or "",
            )

        # Root store: settings with no parent
        root = Gio.ListStore(item_type=ASLSetting)
        for name, meta in self.settings.items():
            if not (meta.get("parent") or ""):
                root.append(items[name])

        def create_func(item: ASLSetting):
            parent_name = item.setting_name
            children = Gio.ListStore(item_type=ASLSetting)
            for name, meta in self.settings.items():
                if (meta.get("parent") or "") == parent_name:
                    children.append(items[name])
            # Return None when no children so expanders render correctly
            return children if children.get_n_items() > 0 else None

        # passthrough=True so rows expose our ASLSetting items directly
        tree = Gtk.TreeListModel.new(root, True, False, create_func)
        return tree


class ASLInterpreter(GObject.Object):
    __gtype__name__ = "ASLInterpreter"

    def __init__(self, asl_script: str):
        super().__init__()
        self.asl_path = asl_script
        self.asl_script = open(asl_script, "r", encoding="utf-8").read().splitlines()
        # region State Declarations
        self.states = self.find_states()
        # endregion
        # region Runtime State
        self.vars = {}
        self.org = {}
        self.current = {}
        self.old = {}
        self.version = "Unknown"
        self.settings = ASLSettings()
        self.brace_loc = 0
        self.plugin = None
        self.initialized = False
        self._early_exit_update = False
        # endregion
        # region Initialization
        self.startup()
        self.modules = ASLModule()
        self.org = self.vars.copy()
        self.exit = self.exit_func
        self.update = self.update_func
        # Load a game-specific plugin if available (based on first state's process name)
        first_proc = None
        if self.states:
            first_proc = self.states[0].process_name
        if first_proc:
            self.plugin = load_plugin(first_proc)
            if self.plugin:
                try:
                    self.plugin.setup()
                except Exception:
                    self.plugin = None

        GLib.timeout_add(50.0 / 3.0, self.state_update)

    def update_func(self):
        # Find and isolate the 'update' block, then execute only its inner lines.
        self.i = 0
        n = len(self.asl_script)
        update_vars = {}

        def extract_block_after(idx: int) -> tuple[int, list[str]]:
            i = idx
            # Move to the first '{'
            while i < n and "{" not in self.asl_script[i]:
                i += 1
            if i >= n:
                return i, []
            # Collect balanced block starting from the line with '{'
            depth = 0
            block: list[str] = []
            while i < n:
                line = self.asl_script[i]
                depth += line.count("{")
                block.append(line)
                i += 1
                depth -= line.count("}")
                if depth <= 0:
                    break
            # inner between first '{' and its matching '}'
            start_in = 0
            for k, s in enumerate(block):
                if "{" in s:
                    start_in = k + 1
                    break
            end_in = max(start_in, len(block) - 1)
            inner = block[start_in:end_in]
            return i, inner

        while self.i < n:
            line = self.asl_script[self.i].strip()
            if line.startswith("update"):
                next_i, inner_lines = extract_block_after(self.i + 1)
                # Execute only inside the update block
                self._exec_lines(inner_lines, update_vars)
                # Move index to end of the block and stop
                self.i = next_i
                break
            self.i += 1

    def state_update(self):
        for state in self.states:
            if not self.modules.process and ASLModule(state.process_name).process:
                self.modules = ASLModule(state.process_name)
            if state.version == self.version:
                for var_name, meta in state.variables.items():
                    try:
                        val = find_variable_value(
                            state.process_name,
                            meta.get("base_off", 0),
                            meta.get("offsets", []),
                            meta.get("base_module", "") or "",
                            meta.get("type", "string256"),
                        )
                        if val != self.old.get(var_name) and var_name in self.current:
                            self.old[var_name] = self.current[var_name]
                        self.current[var_name] = val
                    except Exception as e:
                        self.current[var_name] = None

        if self.initialized:
            if self._early_exit_update:
                self._early_exit_update = False
                return True
            self.update_func()
            # Allow game plugin to compute additional dynamic values
            if self.plugin:
                try:
                    self.plugin.update(self)
                except Exception:
                    pass
        else:
            self.initialize()

        # Returning True keeps the GLib timeout running
        return True

    def initialize(self):
        if not self.modules.process:
            return
        self.i = 0
        closing_brace = 0
        init_vars = {}
        n = len(self.asl_script)
        self.hash_switch = []
        while self.i < n:
            line = self.asl_script[self.i].strip()
            if line.startswith("init"):
                self.i += 1
                while self.i < n and "{" not in self.asl_script[self.i]:
                    self.i += 1
                if self.i >= n:
                    break
                self.i += 1
                while self.i < n:
                    if self.asl_script[self.i].startswith("}"):
                        break
                    s = self.asl_script[self.i].strip()
                    # Capture switch(hash) block for version mapping
                    if s.startswith("switch(hash)"):
                        while self.i < n and "}" not in self.asl_script[self.i]:
                            self.hash_switch.append(self.asl_script[self.i])
                            self.i += 1
                        # do not advance here; outer loop will increment
                    else:
                        local_store = self._process_simple_statement_line(s, init_vars)
                        if local_store:
                            init_vars.update(local_store)
                    self.i += 1
                break
            self.i += 1
        self.hash = self.plugin.initialize(self) if self.plugin else None
        if self.hash:
            self.version = self.hash_match(self.hash)
        self.initialized = True

    def hash_match(self, hash: str):
        if not self.hash_switch:
            return None
        self.i = 0
        for line in self.hash_switch:
            line = line.strip()
            mc = re.match(r'^case\s+"([^"]+)"\s*:.*$', line)
            if mc and mc.group(1) == hash:
                # Next line should be return statement
                while not line.strip().startswith("version =") and self.i < len(
                    self.hash_switch
                ):
                    self.i += 1
                    line = self.hash_switch[self.i].strip()
                if self.i < len(self.hash_switch):
                    ret_line = self.hash_switch[self.i].strip()
                    mr = re.match(r"^version =\s+(.*);\s*$", ret_line)
                    if mr:
                        expr = mr.group(1).strip()
                        result = self.identify_type(expr)
                        print(f"Hash match {hash} -> {result}")
                        return result
            self.i += 1
        return None
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

    def reset_vars(self):
        self.vars = self.org.copy()

    # ---- Mini statement & control-flow helpers ----
    def _eval_condition(self, cond: str) -> bool:
        py = self._translate_expr(cond)

        class _NS:
            def __init__(self, d: dict, prefix: str | None = None):
                self._d = d
                self._p = prefix

            def __getattr__(self, name):
                if name in self._d:
                    return self._d.get(name)
                if self._p and f"{self._p}.{name}" in self._d:
                    return self._d.get(f"{self._p}.{name}")
                return None

        env = {
            "version": self.version,
            "current": _NS(self.current, "current"),
            "old": _NS(self.old, "old"),
            "vars": _NS(self.vars, "vars"),
        }
        try:
            return bool(
                eval(compile(py, "<asl-if>", "eval"), {"__builtins__": {}}, env)
            )
        except Exception:
            return False

    def _read_block_or_single(self, start_idx: int) -> tuple[int, list[str]]:
        """Return (next_index, inner_lines) for a block or a single statement.

        - Skips blank lines starting at start_idx.
        - If a '{' is on the current line OR the previous non-empty line,
          collect a balanced brace block and return its inner lines.
        - Otherwise, return the next line as a single statement.
        """
        i = start_idx
        n = len(self.asl_script)
        while i < n and self.asl_script[i].strip() == "":
            i += 1
        if i >= n:
            return i, []

        # Detect if we are inside a block whose '{' was on the previous line
        prev_idx = i - 1
        while prev_idx >= 0 and self.asl_script[prev_idx].strip() == "":
            prev_idx -= 1
        prev_has_open = prev_idx >= 0 and ("{" in self.asl_script[prev_idx])

        # If current line has '{' or previous line opened a block, read a block
        if prev_has_open or ("{" in self.asl_script[i]):
            depth = 1 if (prev_has_open and "{" not in self.asl_script[i]) else 0
            block: list[str] = []
            while i < n:
                line = self.asl_script[i]
                depth += line.count("{")
                depth -= line.count("}")
                block.append(line)
                i += 1
                if depth <= 0:
                    break

            # Compute inner lines between the opening '{' and its matching '}'
            start_in_block = 0
            if not prev_has_open:
                # Skip up to and including the first '{' on/after current line
                for k, s in enumerate(block):
                    if "{" in s:
                        start_in_block = k + 1
                        break
            end_in_block = max(start_in_block, len(block) - 1)  # drop closing line
            inner = block[start_in_block:end_in_block]
            return i, inner

        # single statement
        return i + 1, [self.asl_script[i]]

    def _consume_block_from_list(
        self, lines: list[str], pos: int
    ) -> tuple[int, list[str]]:
        """From a local list of lines, return (next_index, inner_lines) for a block starting at pos.
        If a '{' appears at or after pos, read a balanced block and return its inner lines;
        otherwise return the single following line.
        """
        i2 = pos
        n2 = len(lines)
        while i2 < n2 and lines[i2].strip() == "":
            i2 += 1
        if i2 >= n2:
            return i2, []
        if "{" in lines[i2]:
            depth2 = 0
            block2: list[str] = []
            while i2 < n2:
                ln = lines[i2]
                depth2 += ln.count("{")
                depth2 -= ln.count("}")
                block2.append(ln)
                i2 += 1
                if depth2 <= 0:
                    break
            # inner between first '{' and closing '}'
            start_in = 0
            for k2, s2 in enumerate(block2):
                if "{" in s2:
                    start_in = k2 + 1
                    break
            end_in = max(start_in, len(block2) - 1)
            return i2, block2[start_in:end_in]
        # single statement
        return i2 + 1, [lines[i2]]

    def _exec_lines(self, lines: list[str], local_store: dict | None):
        """Execute a list of simple statements, supporting nested if/else inside the list."""
        k = 0
        while k < len(lines):
            s = lines[k].strip()
            if not s:
                k += 1
                continue
            if s.startswith("if"):
                mloc = re.match(r"^if\s*\((.*)\)\s*$", s) or re.match(
                    r"^if\s*\((.*)\)\s*\{?\s*$", s
                )
                if not mloc:
                    k += 1
                    continue
                cond_loc = mloc.group(1).strip()
                k_then_end, then_lines_loc = self._consume_block_from_list(lines, k + 1)
                # else / else if
                j2 = k_then_end
                while j2 < len(lines) and lines[j2].strip() == "":
                    j2 += 1
                else_lines_loc: list[str] = []
                elif_cond_loc: str | None = None
                if j2 < len(lines) and lines[j2].strip().startswith("else"):
                    token2 = lines[j2].strip()
                    # Allow optional '{' and comments in nested else-if
                    m2loc = re.match(
                        r"^else\s+if\s*\((.*)\)\s*(?:\{)?\s*(?://.*)?$",
                        token2,
                    )
                    j2 += 1
                    if m2loc:
                        elif_cond_loc = m2loc.group(1).strip()
                        j2, else_lines_loc = self._consume_block_from_list(lines, j2)
                    else:
                        j2, else_lines_loc = self._consume_block_from_list(lines, j2)

                run_then_loc = self._eval_condition(cond_loc)
                run_else_loc = False
                if not run_then_loc and elif_cond_loc is not None:
                    run_else_loc = self._eval_condition(elif_cond_loc)
                elif not run_then_loc and else_lines_loc:
                    run_else_loc = True

                chosen = (
                    then_lines_loc
                    if run_then_loc
                    else (else_lines_loc if run_else_loc else [])
                )
                self._exec_lines(chosen, local_store)
                k = j2
                if getattr(self, "_early_exit_update", False):
                    return
                continue
            if s == "return false;":
                self._early_exit_update = True
                return
            result = self._process_simple_statement_line(s, local_store)
            if result:
                local_store.update(result)
            k += 1

    def _process_if_block_in_stream(self, local_store: dict | None) -> dict:
        """Process an if (...) [then] [else] at self.i; advances self.i past it."""
        n = len(self.asl_script)
        header = self.asl_script[self.i].strip()
        m = re.match(r"^if\s*\((.*)\)\s*$", header)
        if not m:
            m = re.match(r"^if\s*\((.*)\)\s*\{?\s*$", header)
        if not m:
            self.i += 1
            return
        cond = m.group(1).strip()
        # Move to then body
        self.i += 1
        then_end, then_lines = self._read_block_or_single(self.i)
        # Check for else/else if
        j = then_end
        while j < n and self.asl_script[j].strip() == "":
            j += 1
        else_lines: list[str] = []
        elif_cond: str | None = None
        if j < n and self.asl_script[j].strip().startswith("else"):
            token = self.asl_script[j].strip()
            # Allow optional trailing '{' and end-of-line comments after else-if
            m2 = re.match(r"^else\s+if\s*\((.*)\)\s*(?:\{)?\s*(?://.*)?$", token)
            j += 1
            if m2:
                elif_cond = m2.group(1).strip()
                j, else_lines = self._read_block_or_single(j)
            else:
                j, else_lines = self._read_block_or_single(j)

        # Decide branch
        run_then = self._eval_condition(cond)
        run_else = False
        if not run_then and elif_cond is not None:
            run_else = self._eval_condition(elif_cond)
        elif not run_then and else_lines:
            run_else = True

        # Execute the chosen simple lines (supports nested ifs within the extracted list)
        target = then_lines if run_then else (else_lines if run_else else [])
        self._exec_lines(target, local_store)

        # Advance after processed blocks
        self.i = j
        if getattr(self, "_early_exit_update", False):
            # Skip to end of update block
            while self.i < n and not self.asl_script[self.i].strip().startswith("}"):
                self.i += 1

        return local_store

    def _process_simple_statement_line(
        self, s: str, local_store: dict | None = None
    ) -> dict:
        """Handle very simple one-line statements used in ASL blocks:
        - assignments to vars.* and current.*
        - local 'var name = expr;' captures into provided local_store
        - calls to vars.resetVars()
        Ignores anything else.
        """
        if not s:
            return
        # Local var declarations
        if s.startswith("var ") and "=" in s and s.endswith(";"):
            name = s.split("=", 1)[0].strip().split(" ", 1)[1].strip()
            rhs = s.split("=", 1)[1].strip().strip(";")
            val = self.identify_type(rhs)
            if local_store is not None:
                local_store[name] = val
                return local_store
            return None
        # Assignments to vars.* or current.*
        if "=" in s and s.endswith(";"):
            lhs, rhs = s.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip().strip(";")
            if lhs.startswith("vars."):
                lhs = lhs.replace("vars.", "", 1)
                self.vars[lhs] = self.identify_type(rhs)
                return
            if lhs.startswith("current."):
                lhs = lhs.replace("current", "", 1)
                self.current[lhs] = self.identify_type(rhs)
                return None
        # Reset call
        if "vars.resetVars()" in s:
            self.reset_vars()
            return None

    def exit_func(self):
        self.i = 0
        n = len(self.asl_script)
        exit_vars = {}
        while self.i < n:
            line = self.asl_script[self.i].strip()
            if line.startswith("exit"):
                # Advance to block start '{'
                self.i += 1
                while self.i < n and "{" not in self.asl_script[self.i]:
                    self.i += 1
                if self.i >= n:
                    break
                # Inside block until matching '}' on a line
                self.i += 1
                while self.i < n:
                    s = self.asl_script[self.i].strip()
                    if s.startswith("}"):
                        break
                    local_store = self._process_simple_statement_line(s, None)
                    if local_store:
                        exit_vars.update(local_store)
                    self.i += 1
                break
            self.i += 1

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
                key = m[1]
                params = [p.strip() for p in m[2].split(",") if p.strip()]
                expr = m[3]
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
    def startup(self):
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
                    self.asl_script[self.i]
                    .strip()
                    .strip(";")
                    .split("=")[0]
                    .strip()
                    .replace("vars.", "")
                )
                var_value = self.identify_type(
                    self.asl_script[self.i].strip().strip(";").split("=")[1]
                )
                self.vars[var_name] = var_value
            self.i += 1
        # Fallback/global parsing to ensure settings exist before tooltips
        self._parse_settings_and_tooltips_global()

    # endregion

    def identify_type(self, value: str):
        if value.startswith("//"):
            return "INVALID"
        value = value.strip()
        n = len(self.asl_script)
        if "//" in value:
            value = value.split("//")[0]
        if ";" in value:
            value = value.strip().strip(";")
        # Normalize C# verbatim strings @"..." to safe Python strings
        value = self._replace_verbatim_strings(value)

        # Helper: evaluate a very small subset of expressions used in the ASL
        # for paths (properties/functions and string concatenations).
        def _eval_expr(expr: str):
            s = expr.strip()
            # Quick-out quoted strings
            if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return s.strip('"')

            # module.FileName -> full path to the game module
            if s == "module.FileName":
                try:
                    if self.modules and getattr(self.modules, "process", None):
                        m = self.modules.First()
                        if isinstance(m, dict) and "path" in m:
                            return m["path"]
                except Exception:
                    pass
                return ""

            # new FileInfo(x).DirectoryName => Path(x).parent
            m = re.match(r"^new\s+FileInfo\s*\((.*)\)\.DirectoryName$", s)
            if m:
                inner = _eval_expr(m.group(1))
                try:
                    return str(Path(str(inner)).parent)
                except Exception:
                    return ""

            # Bare new FileInfo(x) => Path(x) string representation
            m = re.match(r"^new\s+FileInfo\s*\((.*)\)$", s)
            if m:
                inner = _eval_expr(m.group(1))
                try:
                    return str(Path(str(inner)))
                except Exception:
                    return str(inner)

            # Simple string concatenation A + B (used for paths)
            # Split only at top-level '+', not inside quotes (already handled) or parens.
            depth = 0
            plus_pos = -1
            for i, ch in enumerate(s):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth = max(0, depth - 1)
                elif ch == "+" and depth == 0:
                    plus_pos = i
                    break
            if plus_pos != -1:
                left = s[:plus_pos]
                right = s[plus_pos + 1 :]
                lval = _eval_expr(left)
                rval = _eval_expr(right)
                # Ensure string concatenation
                return f"{lval}{rval}"

            return s

        if "new " in value:
            if value.startswith("new Dictionary"):
                return self._parse_dictionary_of_funcs()
            if value.startswith("new HashSet"):
                return set()
            # Map new FileInfo(x).DirectoryName and similar property chains via _eval_expr
            if value.startswith("new FileInfo"):
                # Try to evaluate full expression (possibly with + "..." appended)
                try:
                    return _eval_expr(value)
                except Exception:
                    pass
                return ""
            return "INVALID"
        if "modules." in value and self.modules.process:
            if value.strip() == "modules.First()":
                return self.modules.First()
            if value.strip() == "modules.BaseAddress":
                return self.modules.BaseAddress
            if value.strip() == "modules.is64Bit()":
                return self.modules.is64Bit()

        # Support property lookups used directly in expressions, e.g., module.FileName
        if value == "module.FileName":
            try:
                if self.modules and getattr(self.modules, "process", None):
                    m = self.modules.First()
                    if isinstance(m, dict) and "path" in m:
                        return m["path"]
            except Exception:
                pass
            return ""

        # Evaluate simple concatenations and property chains for strings/paths
        if any(tok in value for tok in ("+", ".DirectoryName", "module.FileName")):
            try:
                return _eval_expr(value)
            except Exception:
                pass
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
                    target.replace("vars.", "", 1)

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
        """Global regex-based parsing for settings.Add/SetToolTip and
        settings.CurrentDefaultParent in source order.

        - Respects explicit parent in Add(id, default, desc, parent)
        - Applies settings.CurrentDefaultParent when parent is omitted
        - Handles tooltip strings and verbatim string concatenation
        """
        text = "\n".join(self.asl_script)

        pattern = re.compile(
            r"""
            (?P<defparent>settings\.CurrentDefaultParent\s*=\s*(?P<defval>null|\"[^\"]*\")\s*;) |
            (?P<add>settings\.Add\s*\(\s*\"(?P<add_name>[^\"]+)\"\s*
                (?:,\s*(?P<add_bool>true|false))?\s*
                (?:,\s*\"(?P<add_desc>[^\"]*)\")?\s*
                (?:,\s*(?P<add_parent>null|\"[^\"]*\"))?\s*\)\s*;) |
            (?P<tip>settings\.SetToolTip\s*\(\s*\"(?P<tip_name>[^\"]+)\"\s*,\s*(?P<tip_expr>.*?)\)\s*;)
            """,
            re.I | re.S | re.X,
        )

        current_parent: str | None = None

        for m in pattern.finditer(text):
            # Handle CurrentDefaultParent assignment
            if m.group("defparent"):
                raw = m.group("defval")
                if raw is None or raw.lower() == "null":
                    current_parent = None
                else:
                    # strip quotes
                    current_parent = raw.strip()[1:-1]
                continue

            # Handle settings.Add
            if m.group("add"):
                name = m.group("add_name")
                val = (m.group("add_bool") or "true").lower()
                desc = m.group("add_desc") or ""
                parent_tok = m.group("add_parent")
                if parent_tok is None:
                    parent = current_parent
                else:
                    if parent_tok.lower() == "null":
                        parent = None
                    else:
                        parent = parent_tok.strip()[1:-1]
                default = True if val == "true" else False
                try:
                    self.settings.add(name, default, desc, parent)
                except Exception:
                    pass
                continue

            # Handle tooltips
            if m.group("tip"):
                name = m.group("tip_name")
                expr = self._replace_verbatim_strings(m.group("tip_expr"))
                tooltip = self._eval_const_string_concat(expr)
                if isinstance(tooltip, str):
                    try:
                        self.settings.set_tooltip(name, tooltip)
                    except Exception:
                        pass
                continue

    # endregion


if __name__ == "__main__":
    asl = ASLInterpreter("Deltarune.asl")
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt as e:
        print("Exiting...")
