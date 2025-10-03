# region Imports
import math
import time
import re
import ast
import threading
from datetime import datetime, timedelta
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import importlib.resources
from types import SimpleNamespace
from gi.repository import GObject, GLib, Gio, Gtk

try:
    from aslrunner.proc_utils import (
        find_variable_value,
        find_wine_process,
        get_all_modules,
        is_64_bit,
        get_module_base,
        get_module_memory,
    )
    from aslrunner.game_plugins import load_plugin
except ModuleNotFoundError:
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


class TimerPhase:
    NotRunning = 0
    Running = 1
    Paused = 2
    Ended = 3


class TimingMethod:
    RealTime = 0
    GameTime = 1


class Run:
    Offset = 0


class DummyTimer:
    """Lightweight stand-in for LiveSplit's timer interface used in ASL.

    Only holds data fields that ASL scripts may read/set. No behavior.
    """

    def __init__(self):
        # Attempt timing
        self.AttemptStarted: datetime | None = None
        self.AttemptEnded: datetime | None = None

        # Start timestamps
        self.AdjustedStartTime: datetime | None = None
        self.StartTimeWithOffset: datetime | None = None
        self.StartTime: datetime | None = None

        # Pause tracking
        self.TimePausedAt: timedelta = timedelta(0)
        self.GameTimePauseTime: timedelta | None = None
        self.IsGameTimePaused: bool = False

        # State/config
        self.CurrentPhase: int = TimerPhase.NotRunning
        self.CurrentComparison: str = "Current Comparison"
        self.CurrentTimingMethod: int = TimingMethod.RealTime
        self.CurrentHotkeyProfile: str = "Default"

        class _Run:
            def __init__(self):
                self.Offset: timedelta = timedelta(0)

        self.Run = _Run()


class _Stopwatch:
    def __init__(self):
        self._start: float | None = None
        self._elapsed: float = 0.0

    def Start(self):
        if self._start is None:
            self._start = time.monotonic()

    def Stop(self):
        if self._start is not None:
            self._elapsed += time.monotonic() - self._start
            self._start = None

    def Reset(self):
        self._elapsed = 0.0
        self._start = None

    @property
    def IsRunning(self) -> bool:
        return self._start is not None

    @property
    def ElapsedMilliseconds(self) -> int:
        total = self._elapsed
        if self._start is not None:
            total += time.monotonic() - self._start
        return int(total * 1000)


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
        try:
            mods = get_all_modules(self.process["pid"]) if self.process else []
        except Exception:
            mods = []
        return mods[0] if mods else None

    def is64Bit(self):
        return is_64_bit(self.process["pid"]) if self.process else False

    # Alias PascalCase name expected by ASL scripts
    Is64Bit = is64Bit


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
    tooltip = GObject.Property(type=str, default="")

    def __init__(
        self,
        setting_name,
        display_name,
        setting_value=False,
        parent="",
        tooltip="",
    ):
        super().__init__()
        self.setting_name = setting_name
        self.display_name = display_name
        self.setting_value = setting_value
        self.parent = parent
        self.tooltip = tooltip


class ASLSettings:

    def __init__(self):
        super().__init__()
        self.settings = {}
        self.start = True
        self.reset = True
        self.split = True

    def add(self, setting_name, setting_value, display_name, parent=""):
        self.settings[setting_name] = {
            "value": setting_value,
            "default": setting_value,
            "display_name": display_name,
            "parent": parent or None,
        }

    def set_tooltip(self, setting_name, tooltip):
        if setting_name in self.settings:
            self.settings[setting_name]["tooltip"] = tooltip

    def export_list_object(self, *, passthrough: bool = True):
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
                tooltip=meta.get("tooltip", ""),
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
        tree = Gtk.TreeListModel.new(root, passthrough, False, create_func)
        return tree


class ASLInterpreter(GObject.Object):
    __gtype__name__ = "ASLInterpreter"

    _SAFE_EVAL_NAMES = {
        "int": int,
        "float": float,
        "bool": bool,
        "str": str,
        "abs": abs,
        "max": max,
        "min": min,
        "round": round,
        "len": len,
    }

    _FLOW_CONTINUE = object()
    _FLOW_BREAK = object()

    split_signal = GObject.Signal("split_signal")
    reset_signal = GObject.Signal("reset_signal")
    start_signal = GObject.Signal("start_signal")
    pause_signal = GObject.Signal("pause_signal")
    initialized_signal = GObject.Signal("initialized")

    def __init__(
        self, game_name: str = "", asl_script: str = "", asl_settings: dict = {}
    ):
        super().__init__()
        if asl_script is None or not os.path.exists(asl_script):
            if game_name:
                asl_script = str(
                    importlib.resources.files("asl_scripts").joinpath(
                        f"{game_name}.asl"
                    )
                )

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
        self._action_return_set = False
        self._action_return = None
        # Lines from the ASL update block a plugin has already handled this tick
        self._skip_update_norms: set[str] = set()
        # Expose a dummy timer for ASL access
        self.timer = DummyTimer()
        self._has_started = False
        # endregion
        # region Initialization
        self.modules = ASLModule()
        self.startup()
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
        if asl_settings:
            for key, val in asl_settings.items():
                if key in self.settings.settings:
                    self.settings.settings[key]["value"] = val

        self._tick_interval = 1.0 / 60.0
        self._worker_stop = threading.Event()
        self._worker_thread = None

    def _emit_signal_async(self, signal_name: str):
        def _forward():
            GObject.Object.emit(self, signal_name)
            return False

        GLib.idle_add(_forward, priority=GLib.PRIORITY_DEFAULT)

    def _run_worker_loop(self):
        tick = self._tick_interval
        while not self._worker_stop.is_set():
            start = time.perf_counter()
            try:
                self.state_update()
            except Exception as exc:
                print(f"ASLInterpreter state update failed: {exc}")
            elapsed = time.perf_counter() - start
            remaining = tick - elapsed
            if remaining > 0:
                self._worker_stop.wait(remaining)

    def start_runtime(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(
            target=self._run_worker_loop,
            name="ASLInterpreterLoop",
            daemon=True,
        )
        self._worker_thread.start()

    def stop_runtime(self, join: bool = False):
        self._worker_stop.set()
        thread = self._worker_thread
        self._worker_thread = None
        if join and thread and thread.is_alive():
            thread.join(timeout=0.5)

    def _extract_block(self, header: str) -> list[str]:
        """Return inner lines of the first block with the given header token, else []."""
        self.i = 0
        n = len(self.asl_script)
        # Find header
        while self.i < n:
            line = self.asl_script[self.i].strip()
            if line == header:
                break
            self.i += 1
        if self.i >= n:
            return []
        # Move to first '{'
        self.i += 1
        while self.i < n and "{" not in self.asl_script[self.i]:
            self.i += 1
        if self.i >= n:
            return []
        # Collect balanced block
        depth = 0
        block: list[str] = []
        while self.i < n:
            line = self.asl_script[self.i]
            depth += line.count("{")
            block.append(line)
            self.i += 1
            depth -= line.count("}")
            if depth <= 0:
                break
        # Extract inner
        start_in = 0
        for k, s in enumerate(block):
            if "{" in s:
                start_in = k + 1
                break
        end_in = max(start_in, len(block) - 1)
        return block[start_in:end_in]

    class _SettingsProxy:
        def __init__(self, settings: "ASLSettings"):
            self._settings = settings

        def __getitem__(self, key: str):
            try:
                meta = self._settings.settings.get(key)
                if meta is None:
                    return False
                return bool(meta.get("value", False))
            except Exception:
                return False

    def update_func(self) -> bool:
        """Run the update block and return False if it returns false; else True.

        Returning False indicates subsequent timer-control actions should be skipped,
        consistent with ASL documentation.
        """
        self._action_return_set = False
        self._action_return = None
        inner_lines = self._extract_block("update")
        if not inner_lines:
            return True
        # If a plugin reported some lines as already handled, skip them
        if self._skip_update_norms:

            def _norm(s: str) -> str:
                return re.sub(r"\s+", "", s.strip())

            filtered = [
                ln for ln in inner_lines if _norm(ln) not in self._skip_update_norms
            ]
        else:
            filtered = inner_lines

        self._exec_lines(filtered, {})
        # If update explicitly returned a boolean, use it, else default to True
        if self._action_return_set:
            return bool(self._action_return)
        return True

    def start_func(self) -> bool:
        """Run the start block and return True if it requests a start."""
        self._action_return_set = False
        self._action_return = None
        inner_lines = self._extract_block("start")
        if not inner_lines:
            return False
        self._exec_lines(inner_lines, {})
        return bool(self._action_return) if self._action_return_set else False

    def is_loading_func(self) -> bool:
        self._action_return_set = False
        self._action_return = None
        lines = self._extract_block("isLoading")
        if not lines:
            return False
        self._exec_lines(lines, {})
        return bool(self._action_return) if self._action_return_set else False

    def reset_func(self) -> bool:
        self._action_return_set = False
        self._action_return = None
        lines = self._extract_block("reset")
        if not lines:
            return False
        self._exec_lines(lines, {})
        return bool(self._action_return) if self._action_return_set else False

    def split_func(self) -> bool:
        self._action_return_set = False
        self._action_return = None
        lines = self._extract_block("split")
        if not lines:
            return False
        self._exec_lines(lines, {})
        return bool(self._action_return) if self._action_return_set else False

    def game_time_func(self):
        self._action_return_set = False
        self._action_return = None
        lines = self._extract_block("gameTime")
        if not lines:
            return None
        self._exec_lines(lines, {})
        return self._action_return if self._action_return_set else None

    def on_start_func(self):
        lines = self._extract_block("onStart")
        if lines:
            self._exec_lines(lines, {})

    def on_reset_func(self):
        lines = self._extract_block("onReset")
        if lines:
            self._exec_lines(lines, {})

    def on_split_func(self):
        lines = self._extract_block("onSplit")
        if lines:
            self._exec_lines(lines, {})

    def state_update(self):
        for state in self.states:
            # Refresh module handle if current PID is invalid or replaced
            try:
                found = find_wine_process(state.process_name)
            except Exception:
                found = []

            if self.modules and getattr(self.modules, "process", None):
                cur_pid = (
                    self.modules.process.get("pid") if self.modules.process else None
                )
                cur_alive = bool(cur_pid) and os.path.exists(f"/proc/{cur_pid}")
                same_in_found = bool(cur_pid) and any(
                    p.get("pid") == cur_pid for p in found
                )
                # If pid not alive or not among matching processes, try to switch
                if not cur_alive or not same_in_found:
                    if found:
                        self.modules = ASLModule(state.process_name)
                        self.initialized = False
                    else:
                        # No matching process currently; clear modules
                        self.modules = ASLModule()
                        self.initialized = False
            else:
                # No modules yet; attach if a matching process exists
                if found:
                    self.modules = ASLModule(state.process_name)
                    self.initialized = False
            if self.version == "Unknown":
                self.initialized = False
                break
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
                        if var_name in self.current and self.current[var_name]:
                            self.old[var_name] = self.current[var_name]
                        self.current[var_name] = val
                    except Exception as e:
                        self.current[var_name] = None

        if self.initialized:
            self.emit("initialized")
            # Run update and honor an explicit 'return false;' in the bloc
            # Allow game plugin to compute additional dynamic values
            if self.plugin:
                try:
                    skipped = self.plugin.update(self)
                    # Accept a list of raw ASL lines that the plugin handled already
                    self._skip_update_norms.clear()
                    if isinstance(skipped, (list, tuple)):
                        # Normalize by stripping whitespace for robust matching
                        for s in skipped:
                            try:
                                key = re.sub(r"\s+", "", str(s).strip())
                                if key:
                                    self._skip_update_norms.add(key)
                            except Exception:
                                continue
                except Exception:
                    pass
            should_continue = self.update_func()
            # Clear per-tick skips after running update
            self._skip_update_norms.clear()
            if should_continue:
                # Order of execution: start (if not started), then while running: isLoading, gameTime, reset/split
                if self.timer.CurrentPhase == TimerPhase.NotRunning:
                    if self.start_func():
                        # Simulate LiveSplit starting the timer
                        now = datetime.now()
                        self.timer.AttemptStarted = now
                        self.timer.StartTime = now
                        self.timer.StartTimeWithOffset = now + self.timer.Run.Offset
                        self.timer.CurrentPhase = TimerPhase.Running
                        self._has_started = True
                        self._emit_signal_async("start_signal")
                        # onStart event
                        self.on_start_func()
                elif self.timer.CurrentPhase == TimerPhase.Running:
                    # isLoading/gameTime
                    is_loading = self.is_loading_func()
                    # When isLoading is true, set game time paused
                    if isinstance(is_loading, bool):
                        self.timer.IsGameTimePaused = is_loading
                        if is_loading:
                            self._emit_signal_async("pause_signal")
                    _gt = self.game_time_func()
                    # reset / split
                    if self.reset_func():
                        # Simulate reset
                        self.timer.CurrentPhase = TimerPhase.NotRunning
                        self.timer.AttemptEnded = datetime.now()
                        self._has_started = False
                        self._emit_signal_async("reset_signal")
                        self.on_reset_func()
                    else:
                        if self.split_func():
                            self._emit_signal_async("split_signal")
                            self.on_split_func()
        else:
            self.initialize()

        # Returning True keeps the GLib timeout running
        for key, value in self.current.items():
            if value:
                self.old[key] = value
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
                        local_store = self._process_simple_statement_line(
                            s, init_vars, self.i
                        )
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

    def _wrap_callable_arg(self, value):
        """Convert nested dict/list arguments so dot access works inside lambdas."""

        if isinstance(value, self.DotDict):
            return value
        if isinstance(value, dict):
            return self.DotDict(
                {k: self._wrap_callable_arg(v) for k, v in value.items()}
            )
        if isinstance(value, list):
            return [self._wrap_callable_arg(v) for v in value]
        return value

    def _translate_expr(self, expr: str) -> str:
        s = expr.strip().rstrip(",")
        # Convert C# verbatim strings @"..." to Python-safe quoted strings
        s = self._replace_verbatim_strings(s)
        # Handle explicit C#-style casts, e.g. (int)x, (double)(a+b)
        s = self._replace_explicit_casts(s)
        # Translate dot lookups on ASL state dictionaries to Python index syntax
        s = self._replace_state_access(s)
        # Protect '!=' during replacements
        s = s.replace("true", "True")
        s = s.replace("false", "False")
        s = s.replace("null", "None")
        s = s.replace("!=", " __NEQ__ ")
        s = s.replace("&&", " and ")
        s = s.replace("||", " or ")
        # logical not: bare '!'
        s = re.sub(r"(?<![=!])!\s*", " not ", s)
        s = s.replace(" __NEQ__ ", " != ")
        # print(...) -> _print(...)
        s = re.sub(r"\bprint\s*\(", "_print(", s)
        # Methods
        s = re.sub(r"\.EndsWith\s*\(", ".endswith(", s)
        s = re.sub(r"\.StartsWith\s*\(", ".startswith(", s)
        # obj.Contains(arg) -> (arg in obj)
        contains_pattern = re.compile(
            r"((?:[A-Za-z_][A-Za-z0-9_]*)(?:\s*(?:\.(?:[A-Za-z_][A-Za-z0-9_]*)|\[[^\]]+\]))*)\s*\.Contains\s*\(([^()]+)\)"
        )

        def _contains_repl(match: re.Match[str]) -> str:
            lhs = match.group(1).strip()
            rhs = match.group(2).strip()
            return f"({rhs} in {lhs})"

        s = contains_pattern.sub(_contains_repl, s)
        # TimeSpan.FromSeconds(x) -> timedelta(seconds=x)
        s = re.sub(r"\bTimeSpan\.FromSeconds\s*\(", "timedelta(seconds=", s)
        # Convert.ToXxx(x) -> python builtins
        s = re.sub(r"\bConvert\.ToInt32\s*\(", "int(", s)
        s = re.sub(r"\bConvert\.ToUInt32\s*\(", "int(", s)
        s = re.sub(r"\bConvert\.ToInt64\s*\(", "int(", s)
        s = re.sub(r"\bConvert\.ToUInt64\s*\(", "int(", s)
        s = re.sub(r"\bConvert\.ToDouble\s*\(", "float(", s)
        s = re.sub(r"\bConvert\.ToSingle\s*\(", "float(", s)
        s = re.sub(r"\bConvert\.ToBoolean\s*\(", "bool(", s)
        s = re.sub(r"\bConvert\.ToString\s*\(", "str(", s)
        # C# null
        s = re.sub(r"==\s*null", " is None", s, flags=re.IGNORECASE)
        s = re.sub(r"!=\s*null", " is not None", s, flags=re.IGNORECASE)
        # Simple object replacements
        s = re.sub(r"\bnew\s+Stopwatch\s*\(\)", "_Stopwatch()", s)
        # Already normalized verbatim strings above
        # Ternary cond ? a : b (single-level)
        if "?" in s and ":" in s:
            s = self._convert_ternary(s)
        return s

    def _replace_state_access(self, s: str) -> str:
        """Convert current.foo -> current["foo"] for special ASL state dicts."""

        targets = {"current", "old", "vars"}
        out: list[str] = []
        i = 0
        n = len(s)
        in_str = False
        str_ch = ""

        while i < n:
            ch = s[i]
            if in_str:
                out.append(ch)
                if ch == str_ch and (i == 0 or s[i - 1] != "\\"):
                    in_str = False
                i += 1
                continue

            if ch in ('"', "'"):
                in_str = True
                str_ch = ch
                out.append(ch)
                i += 1
                continue

            matched = False
            for name in targets:
                name_len = len(name)
                if s.startswith(name, i) and (
                    i == 0 or not s[i - 1].isalnum() and s[i - 1] != "_"
                ):
                    j = i + name_len
                    k = j
                    while k < n and s[k].isspace():
                        k += 1
                    if k < n and s[k] == ".":
                        k += 1
                        while k < n and s[k].isspace():
                            k += 1
                        ident_start = k
                        if k < n and (s[k].isalpha() or s[k] == "_"):
                            k += 1
                            while k < n and (s[k].isalnum() or s[k] == "_"):
                                k += 1
                            ident = s[ident_start:k]
                            out.append(f'{name}["{ident}"]')
                            i = k
                            matched = True
                            break
                if matched:
                    break

            if matched:
                continue

            out.append(ch)
            i += 1

        return "".join(out)

    def _replace_explicit_casts(self, s: str) -> str:
        """Replace C#-style casts like (int)x with Python calls int(x).

        Heuristics:
        - Recognizes numeric and basic types: sbyte, byte, short, ushort, int, uint, long,
          ulong, float, double, bool, string.
        - If the cast is followed by a parenthesized expression, converts (T)(expr) -> T(expr).
        - Otherwise, converts the next primary expression (identifier/property chain with
          optional call/index) into a function argument.
        """
        type_map = {
            "sbyte": "int",
            "byte": "int",
            "short": "int",
            "ushort": "int",
            "int": "int",
            "uint": "int",
            "long": "int",
            "ulong": "int",
            "float": "float",
            "double": "float",
            "bool": "bool",
            "string": "str",
        }

        out = []
        i = 0
        n = len(s)
        in_str = False
        str_ch = ""
        while i < n:
            ch = s[i]
            # Track quoted strings to avoid replacing inside
            if in_str:
                out.append(ch)
                if ch == str_ch and (i == 0 or s[i - 1] != "\\"):
                    in_str = False
                i += 1
                continue
            if ch in ('"', "'"):
                in_str = True
                str_ch = ch
                out.append(ch)
                i += 1
                continue
            if ch == "(":
                # Try to parse a cast type
                j = i + 1
                while j < n and s[j].isspace():
                    j += 1
                k = j
                while k < n and (s[k].isalpha()):
                    k += 1
                tname = s[j:k]
                while k < n and s[k].isspace():
                    k += 1
                if k < n and s[k] == ")" and tname in type_map:
                    # Found a cast. Now parse the following primary expression
                    func = type_map[tname]
                    m = k + 1
                    while m < n and s[m].isspace():
                        m += 1
                    if m < n and s[m] == "(":
                        # Cast of a parenthesized expression: (T)( ... )
                        depth = 0
                        p = m
                        while p < n:
                            if s[p] == "(":
                                depth += 1
                            elif s[p] == ")":
                                depth -= 1
                                if depth == 0:
                                    break
                            p += 1
                        if p < n:
                            inner = s[m + 1 : p]
                            out.append(f"{func}((" + inner + "))")
                            i = p + 1
                            continue
                    # Otherwise capture a primary: identifier/chain with calls and indexing
                    p = m
                    # First part must start with identifier or keyword like current/vars
                    if p < n and (s[p].isalpha() or s[p] == "_"):
                        # Consume identifier and subsequent .ident, [..], (..)
                        while p < n:
                            if s[p].isalnum() or s[p] in "_":
                                p += 1
                                continue
                            if s[p] == ".":
                                p += 1
                                continue
                            if s[p] == "(":  # consume call
                                depth = 0
                                q = p
                                while q < n:
                                    if s[q] == "(":
                                        depth += 1
                                    elif s[q] == ")":
                                        depth -= 1
                                        if depth == 0:
                                            q += 1
                                            break
                                    q += 1
                                p = q
                                continue
                            if s[p] == "[":  # consume indexing
                                depth = 0
                                q = p
                                while q < n:
                                    if s[q] == "[":
                                        depth += 1
                                    elif s[q] == "]":
                                        depth -= 1
                                        if depth == 0:
                                            q += 1
                                            break
                                    q += 1
                                p = q
                                continue
                            break
                        expr = s[m:p]
                        out.append(f"{func}(" + expr + ")")
                        i = p
                        continue
                # Not a cast, just emit '('
                out.append(ch)
                i += 1
                continue
            out.append(ch)
            i += 1
        return "".join(out)

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
    def _eval_condition(self, cond: str, local_store: dict | None = None) -> bool:
        py = self._translate_expr(cond).strip()

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

            def __getitem__(self, name):
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
            # Timer and enums for ASL conditions
            "timer": self.timer,
            "TimerPhase": TimerPhase,
            "TimingMethod": TimingMethod,
            # timedelta for TimeSpan translation
            "timedelta": timedelta,
            # settings indexer
            "settings": self._SettingsProxy(self.settings),
            "_print": self._print,
            "_Stopwatch": _Stopwatch,
        }
        env.update(self._SAFE_EVAL_NAMES)
        if local_store:
            env.update(local_store)
        try:
            return bool(
                eval(compile(py, "<asl-if>", "eval"), {"__builtins__": {}}, env)
            )
        except Exception as e:
            return False

    def _eval_value(
        self,
        expr: str,
        local_store: dict | None = None,
        *,
        _allow_concat: bool = True,
    ):
        """Evaluate an expression in the ASL environment and return the raw value.
        Returns None on failure.
        """

        if local_store:
            simple = expr.strip()
            if simple in local_store:
                return local_store[simple]

        py = self._translate_expr(expr)
        env = {
            "version": self.version,
            "current": self.current,
            "old": self.old,
            "vars": self.vars,
            "timer": self.timer,
            "TimerPhase": TimerPhase,
            "TimingMethod": TimingMethod,
            "timedelta": timedelta,
            "settings": self._SettingsProxy(self.settings),
            "game": self.modules,
            "_print": self._print,
            "_Stopwatch": _Stopwatch,
        }
        env.update(self._SAFE_EVAL_NAMES)
        if local_store:
            env.update(local_store)
        try:
            return eval(compile(py, "<asl-expr>", "eval"), {"__builtins__": {}}, env)
        except TypeError:
            if _allow_concat:
                concat = self._try_string_concat(expr, local_store)
                if concat is not None:
                    return concat
            return None
        except Exception as e:
            if _allow_concat:
                concat = self._try_string_concat(expr, local_store)
                if concat is not None:
                    return concat
            return None

    def _split_top_level_plus(self, expr: str) -> list[str]:
        """Split an expression on top-level '+' operators, ignoring strings/parens."""

        parts: list[str] = []
        depth = 0
        buf: list[str] = []
        in_str = False
        str_ch = ""
        i = 0
        n = len(expr)
        while i < n:
            ch = expr[i]
            if in_str:
                buf.append(ch)
                if ch == str_ch and (i == 0 or expr[i - 1] != "\\"):
                    in_str = False
                i += 1
                continue
            if ch in ('"', "'"):
                in_str = True
                str_ch = ch
                buf.append(ch)
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if depth == 0 and ch == "+":
                parts.append("".join(buf).strip())
                buf = []
                i += 1
                continue
            buf.append(ch)
            i += 1
        if buf:
            parts.append("".join(buf).strip())
        return parts

    def _strip_wrapping_parens(self, expr: str) -> str | None:
        """Return inner expression if expr is fully enclosed in a single pair of parens."""

        s = expr.strip()
        if not s or s[0] != "(" or s[-1] != ")":
            return None

        depth = 0
        in_str = False
        str_ch = ""
        for i, ch in enumerate(s):
            if in_str:
                if ch == str_ch and (i == 0 or s[i - 1] != "\\"):
                    in_str = False
                continue
            if ch in ('"', "'"):
                in_str = True
                str_ch = ch
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    return None
        if depth == 0:
            inner = s[1:-1].strip()
            return inner
        return None

    def _try_string_concat(
        self, expr: str, local_store: dict | None = None
    ) -> str | None:
        """Attempt C#-style string concatenation, coercing parts to strings."""

        parts = self._split_top_level_plus(expr)
        if len(parts) <= 1:
            inner = self._strip_wrapping_parens(expr)
            if inner and inner != expr.strip():
                return self._try_string_concat(inner, local_store)
            return None

        pieces: list[str] = []
        for part in parts:
            if not part:
                continue
            try:
                val = self._eval_value(part, local_store, _allow_concat=False)
            except Exception:
                val = None
            if val is None:
                val = self._try_string_concat(part, local_store)
            if val is None:
                try:
                    val = ast.literal_eval(part)
                except Exception:
                    if local_store and part in local_store:
                        val = local_store[part]
            if val is None:
                val = ""
            pieces.append(val if isinstance(val, str) else str(val))

        return "".join(pieces) if pieces else ""

    def _split_indexer(
        self, expr: str, local_store: dict | None = None
    ) -> tuple[str | None, object | None]:
        """Split base[index] into (base, evaluated index)."""

        s = expr.strip()
        if not s.endswith("]"):
            return s, None

        depth = 0
        for i in range(len(s) - 1, -1, -1):
            ch = s[i]
            if ch == "]":
                depth += 1
            elif ch == "[":
                depth -= 1
                if depth == 0:
                    base = s[:i].strip()
                    index_expr = s[i + 1 : -1].strip()
                    if not base:
                        return None, None
                    index_val = None
                    if index_expr:
                        index_val = self._eval_value(index_expr, local_store)
                        if index_val is None:
                            try:
                                index_val = ast.literal_eval(index_expr)
                            except Exception:
                                if index_expr.isdigit():
                                    index_val = int(index_expr)
                                elif index_expr:
                                    index_val = index_expr
                    return base, index_val
        return s, None

    def _get_index_value(self, container, index):
        """Return container[index] with safety for list/dict."""

        if container is None:
            return None
        try:
            if isinstance(container, list):
                idx = int(index)
                if idx < 0 or idx >= len(container):
                    return None
                return container[idx]
            if isinstance(container, dict):
                return container.get(index)
            return container[index]
        except Exception:
            return None

    def _assign_index(self, container, index, value):
        """Assign container[index] = value, growing lists as needed."""

        if container is None:
            return
        try:
            if isinstance(container, list):
                idx = int(index)
                if idx < 0:
                    return
                while len(container) <= idx:
                    container.append(None)
                container[idx] = value
                return
            if isinstance(container, dict):
                container[index] = value
                return
            container[index] = value
        except Exception:
            return

    def _iter_foreach(self, collection) -> list:
        if collection is None:
            return []
        if isinstance(collection, dict):
            return [SimpleNamespace(Key=k, Value=v) for k, v in collection.items()]
        if hasattr(collection, "items") and callable(collection.items):
            try:
                return [SimpleNamespace(Key=k, Value=v) for k, v in collection.items()]
            except Exception:
                pass
        try:
            iterator = iter(collection)
        except TypeError:
            return []
        wrapped: list = []
        for item in iterator:
            if isinstance(item, SimpleNamespace) and hasattr(item, "Key"):
                wrapped.append(item)
                continue
            if isinstance(item, tuple) and len(item) == 2:
                wrapped.append(SimpleNamespace(Key=item[0], Value=item[1]))
                continue
            wrapped.append(item)
        return wrapped

    def _parse_array_literal(self, expr: str, local_store: dict | None = None) -> list:
        """Parse a C# new[] { ... } literal embedded in a single expression."""

        start = expr.find("{")
        end = expr.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return []
        body = expr[start + 1 : end].strip()
        if not body:
            return []

        parts: list[str] = []
        buf: list[str] = []
        depth = 0
        in_str = False
        str_ch = ""
        i = 0
        while i < len(body):
            ch = body[i]
            if in_str:
                buf.append(ch)
                if ch == str_ch and (i == 0 or body[i - 1] != "\\"):
                    in_str = False
                i += 1
                continue
            if ch in ('"', "'"):
                in_str = True
                str_ch = ch
                buf.append(ch)
                i += 1
                continue
            if ch in "({[":
                depth += 1
            elif ch in ")}]":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
                i += 1
                continue
            buf.append(ch)
            i += 1
        if buf:
            part = "".join(buf).strip()
            if part:
                parts.append(part)

        values: list = []

        def _strip_comment(segment: str) -> str:
            in_str_local = False
            quote_ch = ""
            i2 = 0
            while i2 < len(segment):
                ch2 = segment[i2]
                if in_str_local:
                    if ch2 == quote_ch and (i2 == 0 or segment[i2 - 1] != "\\"):
                        in_str_local = False
                    i2 += 1
                    continue
                if ch2 in ('"', "'"):
                    in_str_local = True
                    quote_ch = ch2
                    i2 += 1
                    continue
                if ch2 == "/" and i2 + 1 < len(segment) and segment[i2 + 1] == "/":
                    return segment[:i2].strip()
                i2 += 1
            return segment.strip()

        for part in parts:
            if not part:
                continue
            part = _strip_comment(part)
            if not part:
                continue
            val = self._eval_value(part, local_store)
            if val is None:
                lower = part.strip().lower()
                if lower == "null":
                    val = None
                elif lower == "true":
                    val = True
                elif lower == "false":
                    val = False
                else:
                    try:
                        val = ast.literal_eval(part)
                    except Exception:
                        val = part.strip('"')
            values.append(val)
        return values

    def _parse_array_literal_from_script(
        self, start_idx: int | None, local_store: dict | None = None
    ) -> list | None:
        if start_idx is None:
            return None
        i = start_idx
        n = len(self.asl_script)
        # Move to first '{'
        while i < n and "{" not in self.asl_script[i]:
            i += 1
        if i >= n:
            return None
        i += 1
        values: list = []
        while i < n:
            line = self.asl_script[i].strip()
            if line.startswith("}"):
                break
            if not line or line.startswith("//"):
                i += 1
                continue
            segment = line
            # Remove inline comments
            j = 0
            in_str = False
            str_ch = ""
            cut = len(segment)
            while j < len(segment):
                ch = segment[j]
                if in_str:
                    if ch == str_ch and (j == 0 or segment[j - 1] != "\\"):
                        in_str = False
                    j += 1
                    continue
                if ch in ('"', "'"):
                    in_str = True
                    str_ch = ch
                    j += 1
                    continue
                if ch == "/" and j + 1 < len(segment) and segment[j + 1] == "/":
                    cut = j
                    break
                j += 1
            segment = segment[:cut].strip().rstrip(",")
            if segment:
                val = self._eval_value(segment, local_store)
                if val is None:
                    lowered = segment.strip().lower()
                    if lowered == "null":
                        val = None
                    elif lowered == "true":
                        val = True
                    elif lowered == "false":
                        val = False
                    else:
                        try:
                            val = ast.literal_eval(segment)
                        except Exception:
                            val = segment.strip('"')
                values.append(val)
            i += 1
        return values

    def _find_statement_index(self, prefix: str) -> int | None:
        for idx, line in enumerate(self.asl_script):
            if line.strip().startswith(prefix):
                return idx
        return None

    def _evaluate_rhs(
        self,
        rhs: str,
        local_store: dict | None = None,
        statement_index: int | None = None,
    ):
        rhs_strip = rhs.strip()
        if rhs_strip.startswith("new[]"):
            arr = self._parse_array_literal_from_script(statement_index, local_store)
            if arr is not None:
                return arr
            return self._parse_array_literal(rhs_strip, local_store)
        val = self._eval_value(rhs, local_store)
        if val is None:
            val = self.identify_type(rhs)
        return val

    def _print(self, *values):
        try:
            msg = " ".join("" if v is None else str(v) for v in values)
        except Exception:
            msg = ""
        try:
            print(msg)
        except Exception:
            pass

    def _collect_statement(self, start_idx: int) -> tuple[str, int]:
        """Collect lines starting at start_idx until a semicolon at brace depth zero."""

        n = len(self.asl_script)
        parts: list[str] = []
        depth = 0
        i = start_idx
        while i < n:
            raw = self.asl_script[i]
            stripped = raw.strip()
            if not parts and not stripped:
                i += 1
                continue
            parts.append(stripped)
            depth += raw.count("{") - raw.count("}")
            if stripped.endswith(";") and depth <= 0:
                break
            i += 1
        combined = " ".join(p for p in parts if p)
        return combined, i

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
        Supports braces on the same line or the previous non-empty line.
        If no block, returns the single following line.
        """
        i2 = pos
        n2 = len(lines)
        while i2 < n2 and lines[i2].strip() == "":
            i2 += 1
        if i2 >= n2:
            return i2, []

        # Check if previous non-empty line opened a block
        prev_idx = i2 - 1
        while prev_idx >= 0 and lines[prev_idx].strip() == "":
            prev_idx -= 1
        prev_has_open = prev_idx >= 0 and ("{" in lines[prev_idx])

        if prev_has_open or ("{" in lines[i2]):
            depth2 = 1 if (prev_has_open and "{" not in lines[i2]) else 0
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
            if not prev_has_open:
                for k2, s2 in enumerate(block2):
                    if "{" in s2:
                        start_in = k2 + 1
                        break
            end_in = max(start_in, len(block2) - 1)
            return i2, block2[start_in:end_in]
        # single statement
        return i2 + 1, [lines[i2]]

    def _line_is_trivia(self, line: str) -> bool:
        """Return True when the line only contains whitespace or a comment."""

        stripped = line.strip()
        if not stripped:
            return True
        if stripped.startswith("//"):
            return True
        if stripped.startswith("/*"):
            return True
        if stripped.startswith("*") and stripped.endswith("*/"):
            return True
        if stripped == "*/":
            return True
        return False

    def _collect_iflike_header(
        self, lines: list[str], idx: int, prefix: str
    ) -> tuple[str, int]:
        """Join consecutive lines forming an if/else-if header (handles multiline conditions)."""

        parts: list[str] = []
        j = idx
        depth = 0
        saw_paren = False

        while j < len(lines):
            raw = lines[j]
            stripped = raw.strip()
            if not parts:
                if not stripped.startswith(prefix):
                    break
            parts.append(stripped)
            code = stripped.split("//", 1)[0]
            if "(" in code:
                saw_paren = True
            depth += code.count("(")
            depth -= code.count(")")
            j += 1
            if saw_paren and depth <= 0:
                break
            # plain else without condition should exit after first line
            if prefix == "else" and not saw_paren:
                break

        return " ".join(parts), j

    def _split_if_condition(self, header: str) -> tuple[str | None, str]:
        """Extract condition and inline suffix from an if/else-if header."""

        code = header.split("//", 1)[0].strip()
        if code.startswith("else if"):
            code = code[4:].strip()
        if not code.startswith("if"):
            return None, ""
        code = code[2:].lstrip()
        if not code.startswith("("):
            return None, ""

        depth = 0
        cond_chars: list[str] = []
        i = 0
        while i < len(code):
            ch = code[i]
            if ch == "(":
                depth += 1
                if depth == 1:
                    i += 1
                    continue
            if ch == ")":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            if depth >= 1:
                cond_chars.append(ch)
            i += 1

        condition = "".join(cond_chars).strip()
        suffix = code[i:].strip()
        return condition, suffix

    def _split_switch_expr(self, header: str) -> tuple[str | None, str]:
        """Extract the selector expression and inline suffix from a switch header."""

        code = header.split("//", 1)[0].strip()
        if not code.startswith("switch"):
            return None, ""
        code = code[6:].lstrip()
        if not code.startswith("("):
            return None, ""

        depth = 0
        expr_chars: list[str] = []
        i = 0
        while i < len(code):
            ch = code[i]
            if ch == "(":
                depth += 1
                if depth == 1:
                    i += 1
                    continue
            if ch == ")":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            if depth >= 1:
                expr_chars.append(ch)
            i += 1

        if depth != 0:
            return None, ""

        expr = "".join(expr_chars).strip()
        suffix = code[i:].strip()
        return (expr if expr else None), suffix

    def _parse_switch_cases(
        self, body_lines: list[str]
    ) -> tuple[list[tuple[list[str], list[str]]], list[str] | None]:
        """Parse switch case bodies into (cases, default)."""

        case_groups: list[tuple[list[str], list[str]]] = []
        default_body: list[str] | None = None
        current_labels: list[str] = []
        current_body: list[str] = []
        current_is_default = False
        current_has_code = False

        def _flush_current():
            nonlocal current_labels, current_body, current_is_default
            nonlocal current_has_code, default_body, case_groups
            if current_is_default:
                default_body = current_body.copy()
            elif current_labels and current_has_code:
                case_groups.append((current_labels.copy(), current_body.copy()))
            current_labels = []
            current_body = []
            current_is_default = False
            current_has_code = False

        for raw in body_lines:
            stripped = raw.strip()
            code = raw.split("//", 1)[0].strip()
            if not code:
                if stripped:
                    current_body.append(raw)
                continue

            m_case = re.match(r"^case\s+(.*?):\s*(.*)$", code)
            if m_case:
                label_expr = m_case.group(1).strip()
                remainder = m_case.group(2).strip()
                if current_labels and not current_is_default and not current_has_code:
                    current_labels.append(label_expr)
                else:
                    _flush_current()
                    current_labels = [label_expr]
                if remainder:
                    current_body.append(remainder)
                    current_has_code = True
                continue

            m_default = re.match(r"^default\s*:\s*(.*)$", code)
            if m_default:
                remainder = m_default.group(1).strip()
                _flush_current()
                current_is_default = True
                if remainder:
                    current_body.append(remainder)
                    current_has_code = True
                continue

            if code.startswith("break"):
                stmt = code if code.endswith(";") else f"{code};"
                current_body.append(stmt)
                current_has_code = True
                _flush_current()
                continue

            current_body.append(raw)
            if code:
                current_has_code = True

        _flush_current()
        return case_groups, default_body

    def _extract_else_inline(self, header: str) -> str:
        """Return inline body (if any) following an else token."""

        code = header.split("//", 1)[0].strip()
        if not code.startswith("else"):
            return ""
        return code[4:].strip()

    def _exec_lines(self, lines: list[str], local_store: dict | None):
        """Execute a list of simple statements, supporting nested if/else inside the list."""
        k = 0
        while k < len(lines):
            s = lines[k].strip()
            if "++" in s:
                print("incrementing")
            if not s:
                k += 1
                continue
            if re.match(r"^continue\s*;?(?:\s*//.*)?$", s):
                return self._FLOW_CONTINUE
            if re.match(r"^break\s*;?(?:\s*//.*)?$", s):
                return self._FLOW_BREAK
            if s.startswith("foreach"):
                mfor = re.match(
                    r"^foreach\s*\(\s*(?:var\s+)?([A-Za-z_][A-Za-z0-9_]*)\s+in\s*(.*)\)\s*(?:\{)?\s*(?://.*)?$",
                    s,
                )
                if not mfor:
                    k += 1
                    continue
                var_name = mfor.group(1)
                iter_expr = mfor.group(2).strip()
                k_body_end, body_lines = self._consume_block_from_list(lines, k + 1)
                iterable = self._eval_value(iter_expr, local_store)
                sequence = self._iter_foreach(iterable)
                had_prev = False
                prev_val = None
                if isinstance(local_store, dict) and var_name in local_store:
                    had_prev = True
                    prev_val = local_store[var_name]
                broke_loop = False
                for item in sequence:
                    if isinstance(local_store, dict):
                        local_store[var_name] = item
                        result = self._exec_lines(body_lines, local_store)
                    else:
                        temp_store = {var_name: item}
                        result = self._exec_lines(body_lines, temp_store)
                    if result is self._FLOW_CONTINUE:
                        continue
                    if result is self._FLOW_BREAK:
                        broke_loop = True
                        break
                    if getattr(self, "_action_return_set", False):
                        broke_loop = True
                        break
                if isinstance(local_store, dict):
                    if had_prev:
                        local_store[var_name] = prev_val
                    else:
                        local_store.pop(var_name, None)
                k = k_body_end
                if getattr(self, "_action_return_set", False):
                    return
                if broke_loop:
                    continue
                continue
            if s.startswith("switch"):
                header, header_end = self._collect_iflike_header(lines, k, "switch")
                switch_expr, inline_suffix = self._split_switch_expr(header)
                if switch_expr is None:
                    k = header_end
                    continue

                inline_suffix = inline_suffix.strip()
                block_start = header_end
                if "{" in inline_suffix:
                    block_start = max(k, header_end - 1)
                k_body_end, body_lines = self._consume_block_from_list(
                    lines, block_start
                )
                if not body_lines:
                    k = k_body_end
                    continue

                case_groups, default_body = self._parse_switch_cases(body_lines)
                selector = self._eval_value(switch_expr, local_store)
                matched = False
                result = None

                for labels, body in case_groups:
                    for label_expr in labels:
                        label_val = self._eval_value(label_expr, local_store)
                        if label_val is None:
                            try:
                                label_val = ast.literal_eval(label_expr)
                            except Exception:
                                label_val = label_expr
                        if selector == label_val:
                            matched = True
                            result = self._exec_lines(body, local_store)
                            if result is self._FLOW_CONTINUE:
                                return result
                            if result is self._FLOW_BREAK:
                                result = None
                            break
                    if matched:
                        break

                if not matched and default_body is not None:
                    result = self._exec_lines(default_body, local_store)
                    if result is self._FLOW_CONTINUE:
                        return result
                    if result is self._FLOW_BREAK:
                        result = None

                k = k_body_end
                if getattr(self, "_action_return_set", False):
                    return
                continue
            if s.startswith("if"):
                header, header_end = self._collect_iflike_header(lines, k, "if")
                cond_loc, inline_suffix = self._split_if_condition(header)
                if cond_loc is None:
                    k = header_end
                    continue

                inline_suffix = inline_suffix.strip()
                if inline_suffix and not inline_suffix.startswith("{"):
                    then_lines_loc = [inline_suffix]
                    k_then_end = header_end
                else:
                    k_then_end, then_lines_loc = self._consume_block_from_list(
                        lines, header_end
                    )
                # Collect zero or more else-if branches and optional final else
                j2 = k_then_end
                while j2 < len(lines) and self._line_is_trivia(lines[j2]):
                    j2 += 1
                elif_branches: list[tuple[str, list[str]]] = []
                else_lines_loc: list[str] | None = None
                while j2 < len(lines):
                    if self._line_is_trivia(lines[j2]):
                        j2 += 1
                        continue
                    token2 = lines[j2].strip()
                    if not token2.startswith("else"):
                        break
                    if token2.startswith("else if"):
                        header2, header2_end = self._collect_iflike_header(
                            lines, j2, "else if"
                        )
                        cond2, suffix2 = self._split_if_condition(header2)
                        j2 = header2_end
                        if cond2 is None:
                            continue
                        suffix2 = suffix2.strip()
                        if suffix2 and not suffix2.startswith("{"):
                            body = [suffix2]
                        else:
                            j2, body = self._consume_block_from_list(lines, j2)
                        elif_branches.append((cond2.strip(), body))
                    else:
                        header_else, header_else_end = self._collect_iflike_header(
                            lines, j2, "else"
                        )
                        inline_else = self._extract_else_inline(header_else).strip()
                        j2 = header_else_end
                        if inline_else and not inline_else.startswith("{"):
                            else_lines_loc = [inline_else]
                        else:
                            j2, else_lines_loc = self._consume_block_from_list(
                                lines, j2
                            )
                        break
                    while j2 < len(lines) and self._line_is_trivia(lines[j2]):
                        j2 += 1

                # Decide branch across then/elif*/else
                chosen: list[str] = []
                if self._eval_condition(cond_loc, local_store):
                    chosen = then_lines_loc
                else:
                    matched = False
                    for cexpr, body in elif_branches:
                        if self._eval_condition(cexpr, local_store):
                            chosen = body
                            matched = True
                            break
                    if not matched and else_lines_loc is not None:
                        chosen = else_lines_loc

                if "tempVar" in ",".join(chosen) and s != "if(current.chapter > 0)":
                    print("found tempVar")
                result = self._exec_lines(chosen, local_store)
                if result is self._FLOW_CONTINUE or result is self._FLOW_BREAK:
                    return result
                k = j2
                if getattr(self, "_action_return_set", False):
                    return
                continue
            if s.startswith("return"):
                # return true/false; or return <expr>; or bare return;
                tok = s.strip().rstrip(";")
                if tok == "return true":
                    self._action_return_set = True
                    self._action_return = True
                    return
                if tok == "return false":
                    self._action_return_set = True
                    self._action_return = False
                    return
                if tok == "return":
                    self._action_return_set = True
                    self._action_return = None
                    return
                # return <expr>
                mret = re.match(r"^return\s+(.*)$", tok)
                if mret:
                    val = self._eval_value(mret.group(1).strip(), local_store)
                    self._action_return_set = True
                    self._action_return = val
                    return
            result = self._process_simple_statement_line(s, local_store)
            if result:
                local_store.update(result)
            k += 1

    def _process_if_block_in_stream(self, local_store: dict | None) -> dict:
        """Process an if (...) [then] [else] at self.i; advances self.i past it."""
        n = len(self.asl_script)
        header, header_end = self._collect_iflike_header(self.asl_script, self.i, "if")
        cond, inline_suffix = self._split_if_condition(header)
        if cond is None:
            self.i = header_end if header_end > self.i else self.i + 1
            return

        inline_suffix = inline_suffix.strip()
        body_idx = header_end
        self.i = body_idx
        if inline_suffix and not inline_suffix.startswith("{"):
            then_lines = [inline_suffix]
            then_end = body_idx
        else:
            then_end, then_lines = self._read_block_or_single(body_idx)
        # Collect zero or more else-if branches and optional final else
        j = then_end
        while j < n and self._line_is_trivia(self.asl_script[j]):
            j += 1
        elif_branches: list[tuple[str, list[str]]] = []
        else_lines: list[str] | None = None
        while j < n:
            if self._line_is_trivia(self.asl_script[j]):
                j += 1
                continue
            token = self.asl_script[j].strip()
            if not token.startswith("else"):
                break
            if token.startswith("else if"):
                header2, header2_end = self._collect_iflike_header(
                    self.asl_script, j, "else if"
                )
                cond2, suffix2 = self._split_if_condition(header2)
                j = header2_end
                if cond2 is None:
                    continue
                suffix2 = suffix2.strip()
                if suffix2 and not suffix2.startswith("{"):
                    body = [suffix2]
                else:
                    j, body = self._read_block_or_single(j)
                elif_branches.append((cond2.strip(), body))
            else:
                header_else, header_else_end = self._collect_iflike_header(
                    self.asl_script, j, "else"
                )
                inline_else = self._extract_else_inline(header_else).strip()
                j = header_else_end
                if inline_else and not inline_else.startswith("{"):
                    else_lines = [inline_else]
                else:
                    j, else_lines = self._read_block_or_single(j)
                break
            while j < n and self._line_is_trivia(self.asl_script[j]):
                j += 1

        # Decide which branch to execute
        target: list[str] = []
        if self._eval_condition(cond, local_store):
            target = then_lines
        else:
            matched = False
            for cexpr, body in elif_branches:
                if self._eval_condition(cexpr, local_store):
                    target = body
                    matched = True
                    break
            if not matched and else_lines is not None:
                target = else_lines

        # Execute the chosen simple lines (supports nested ifs within the extracted list)
        self._exec_lines(target, local_store)

        # Advance after processed blocks
        self.i = j
        if getattr(self, "_early_exit_update", False):
            # Skip to end of update block
            while self.i < n and not self.asl_script[self.i].strip().startswith("}"):
                self.i += 1

        return local_store

    def _process_simple_statement_line(
        self,
        s: str,
        local_store: dict | None = None,
        statement_index: int | None = None,
    ) -> dict:
        """Handle very simple one-line statements used in ASL blocks:
        - assignments to vars.* and current.*
        - local 'var name = expr;' captures into provided local_store
        - calls to vars.resetVars()
        Ignores anything else.
        """
        type_map = {
            "sbyte": "int",
            "byte": "int",
            "short": "int",
            "ushort": "int",
            "int": "int",
            "uint": "int",
            "long": "int",
            "ulong": "int",
            "float": "float",
            "double": "float",
            "bool": "bool",
            "string": "str",
        }
        if not s:
            return
        stripped_stmt = s.strip()
        m_print_stmt = re.match(r"^print\s*\((.*)\)\s*;\s*$", stripped_stmt)
        if m_print_stmt:
            expr = f"print({m_print_stmt.group(1)})"
            try:
                self._eval_value(expr, local_store)
            except Exception:
                pass
            return None

        def _combine_augmented(base, addend):
            if isinstance(base, str) or isinstance(addend, str):
                b = "" if base is None else str(base)
                a = "" if addend is None else str(addend)
                return b + a
            if base is None:
                return addend
            if addend is None:
                return base
            try:
                return base + addend
            except Exception:
                b = "" if base is None else str(base)
                a = "" if addend is None else str(addend)
                return b + a

        def _apply_increment(target: str, amount: int = 1):
            if not target:
                return None
            if target.startswith("vars."):
                target_expr = target.replace("vars.", "", 1)
                if target_expr.endswith("]"):
                    base, index = self._split_indexer(target_expr, local_store)
                    if base and index is not None:
                        container = self.vars.get(base)
                        if container is None:
                            container = [] if isinstance(index, int) else {}
                            self.vars[base] = container
                        current_val = self._get_index_value(container, index)
                        self._assign_index(
                            container,
                            index,
                            _combine_augmented(current_val, amount),
                        )
                        return None
                key = target_expr
                current_val = self.vars.get(key)
                self.vars[key] = _combine_augmented(current_val, amount)
                return None

            if target.startswith("current."):
                target_expr = target.replace("current.", "", 1)
                if target_expr.endswith("]"):
                    base, index = self._split_indexer(target_expr, local_store)
                    if base and index is not None:
                        container = self.current.get(base)
                        if container is None:
                            container = [] if isinstance(index, int) else {}
                            self.current[base] = container
                        current_val = self._get_index_value(container, index)
                        self._assign_index(
                            container,
                            index,
                            _combine_augmented(current_val, amount),
                        )
                        return None
                key = target_expr
                current_val = self.current.get(key)
                self.current[key] = _combine_augmented(current_val, amount)
                return None

            if target.startswith("timer."):
                attr = target.split(".", 1)[1]
                current_val = getattr(self.timer, attr, None)
                try:
                    setattr(
                        self.timer,
                        attr,
                        _combine_augmented(current_val, amount),
                    )
                except Exception:
                    pass
                return None

            if local_store is not None:
                if target.endswith("]"):
                    base, index = self._split_indexer(target, local_store)
                    if base and index is not None:
                        container = local_store.get(base)
                        if container is None:
                            container = [] if isinstance(index, int) else {}
                            local_store[base] = container
                        current_val = self._get_index_value(container, index)
                        self._assign_index(
                            container,
                            index,
                            _combine_augmented(current_val, amount),
                        )
                        return local_store
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", target):
                    current_val = local_store.get(target)
                    local_store[target] = _combine_augmented(current_val, amount)
                    return local_store

            return None

        # Local var declarations
        if s.startswith("var ") and "=" in s and s.endswith(";"):
            name = s.split("=", 1)[0].strip().split(" ", 1)[1].strip()
            rhs = s.split("=", 1)[1].strip().strip(";")
            val = self._eval_value(rhs, local_store)
            if val is None:
                val = self.identify_type(rhs)
            if local_store is not None:
                local_store[name] = val
                return local_store
            return None
        # Assignments to vars.* or current.* or timer.*
        if s.endswith(";"):
            stmt = s.rstrip().rstrip(";")
            if "++" in stmt and "=" not in stmt:
                m_prefix = re.match(r"^\+\+\s*(.+)$", stmt)
                if m_prefix:
                    target = m_prefix.group(1).strip()
                    return _apply_increment(target)
                m_postfix = re.match(r"^(.+?)\s*\+\+$", stmt)
                if m_postfix:
                    target = m_postfix.group(1).strip()
                    return _apply_increment(target)
            if "+=" in stmt:
                lhs, rhs = stmt.split("+=", 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                rhs_val = self._eval_value(rhs, local_store)
                if rhs_val is None:
                    rhs_val = self.identify_type(rhs)

                if lhs.startswith("vars."):
                    target_expr = lhs.replace("vars.", "", 1)
                    if target_expr.endswith("]"):
                        base, index = self._split_indexer(target_expr, local_store)
                        if base and index is not None:
                            container = self.vars.get(base)
                            if container is None:
                                container = [] if isinstance(index, int) else {}
                                self.vars[base] = container
                            current_val = self._get_index_value(container, index)
                            self._assign_index(
                                container,
                                index,
                                _combine_augmented(current_val, rhs_val),
                            )
                            return
                    key = target_expr
                    current_val = self.vars.get(key)
                    self.vars[key] = _combine_augmented(current_val, rhs_val)
                    return
                if lhs.startswith("current."):
                    target_expr = lhs.replace("current.", "", 1)
                    if target_expr.endswith("]"):
                        base, index = self._split_indexer(target_expr, local_store)
                        if base and index is not None:
                            container = self.current.get(base)
                            if container is None:
                                container = [] if isinstance(index, int) else {}
                                self.current[base] = container
                            current_val = self._get_index_value(container, index)
                            new_val = _combine_augmented(current_val, rhs_val)
                            self._assign_index(container, index, new_val)
                            return None
                    key = target_expr
                    current_val = self.current.get(key)
                    new_val = _combine_augmented(current_val, rhs_val)
                    # if key in self.current and self.current.get(key):
                    # self.old[key] = self.current.get(key)
                    self.current[key] = new_val
                    return None
                if lhs.startswith("timer."):
                    attr = lhs.split(".", 1)[1]
                    current_val = getattr(self.timer, attr, None)
                    try:
                        setattr(
                            self.timer,
                            attr,
                            _combine_augmented(current_val, rhs_val),
                        )
                    except Exception:
                        pass
                    return None
                if local_store is not None:
                    if lhs.endswith("]"):
                        base, index = self._split_indexer(lhs, local_store)
                        if base and index is not None:
                            container = local_store.get(base)
                            if container is None:
                                container = [] if isinstance(index, int) else {}
                                local_store[base] = container
                            current_val = self._get_index_value(container, index)
                            self._assign_index(
                                container,
                                index,
                                _combine_augmented(current_val, rhs_val),
                            )
                            return local_store
                    if lhs in local_store:
                        current_val = local_store.get(lhs)
                        local_store[lhs] = _combine_augmented(current_val, rhs_val)
                        return local_store

            if "=" not in stmt:
                return
            lhs, rhs = stmt.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if lhs.startswith("vars."):
                target_expr = lhs.replace("vars.", "", 1)
                # Evaluate dynamic expressions (supports casts); fallback to identify_type
                val = self._evaluate_rhs(rhs, local_store, statement_index)
                if target_expr.endswith("]"):
                    base, index = self._split_indexer(target_expr, local_store)
                    if base and index is not None:
                        container = self.vars.get(base)
                        if container is None:
                            container = [] if isinstance(index, int) else {}
                            self.vars[base] = container
                        self._assign_index(container, index, val)
                        return
                self.vars[target_expr] = val
                return
            if lhs.startswith("current."):
                target_expr = lhs.replace("current.", "", 1)
                new_val = self._evaluate_rhs(rhs, local_store, statement_index)
                if target_expr.endswith("]"):
                    base, index = self._split_indexer(target_expr, local_store)
                    if base and index is not None:
                        container = self.current.get(base)
                        if container is None:
                            container = [] if isinstance(index, int) else {}
                            self.current[base] = container
                        old_val = self._get_index_value(container, index)
                        self._assign_index(container, index, new_val)
                        return None
                key = target_expr
                # if key in self.current and self.current.get(key):
                # self.old[key] = self.current.get(key)
                self.current[key] = new_val
                return None
            if lhs.startswith("timer."):
                attr = lhs.split(".", 1)[1]
                val = self._evaluate_rhs(rhs, local_store, statement_index)
                try:
                    setattr(self.timer, attr, val)
                except Exception:
                    pass
                return None

            if local_store is not None:
                if lhs.endswith("]"):
                    base, index = self._split_indexer(lhs, local_store)
                    if base and index is not None:
                        container = local_store.get(base)
                        if container is None:
                            container = [] if isinstance(index, int) else {}
                            local_store[base] = container
                        val = self._evaluate_rhs(rhs, local_store, statement_index)
                        self._assign_index(container, index, val)
                        return local_store
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", lhs):
                    val = self._evaluate_rhs(rhs, local_store, statement_index)
                    local_store[lhs] = val
                    return local_store

            if lhs.split(" ", 1)[0] in type_map.keys():
                var_type, var_name = lhs.split(" ", 1)
                val = self._evaluate_rhs(rhs, local_store, statement_index)
                local_store[var_name] = val
                return local_store
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
                    local_store = self._process_simple_statement_line(s, None, self.i)
                    if local_store:
                        exit_vars.update(local_store)
                    self.i += 1
                break
            self.i += 1

    def _make_pyfunc(self, params: list[str], expr: str):
        py_expr = self._translate_expr(expr)
        code = compile(py_expr, "<asl-func>", "eval")

        def fn(*args):
            local_env = {
                name: self._wrap_callable_arg(arg) for name, arg in zip(params, args)
            }
            env = {
                "version": self.version,
                "current": self.current,
                "old": self.old,
                "vars": self.vars,
                "settings": self._SettingsProxy(self.settings),
                "timer": self.timer,
                "TimerPhase": TimerPhase,
                "TimingMethod": TimingMethod,
                "timedelta": timedelta,
                "game": self.modules,
                "_print": self._print,
                "_Stopwatch": _Stopwatch,
            }
            env.update(self._SAFE_EVAL_NAMES)
            env.update(local_env)
            return eval(code, {"__builtins__": {}}, env)

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
        n = len(self.asl_script)
        # Capture top-level vars assignments before startup block
        while self.i < n:
            line = self.asl_script[self.i].strip()
            if line.startswith("startup"):
                break
            if line.startswith("vars.") and ("=" in line or "+=" in line):
                stmt, end_idx = self._collect_statement(self.i)
                if stmt:
                    self._process_simple_statement_line(stmt, None, self.i)
                self.i = max(self.i, end_idx) + 1
                continue
            self.i += 1

        # Parse startup block statements
        while self.i < n and not self.asl_script[self.i].strip().startswith("startup"):
            self.i += 1
        if self.i < n and self.asl_script[self.i].strip().startswith("startup"):
            # move to first '{'
            self.i += 1
            while self.i < n and "{" not in self.asl_script[self.i]:
                self.i += 1
            if self.i < n:
                self.i += 1
                while self.i < n:
                    raw = self.asl_script[self.i].strip()
                    if raw.startswith("}"):
                        break
                    if not raw or raw.startswith("//"):
                        self.i += 1
                        continue
                    stmt, end_idx = self._collect_statement(self.i)
                    if stmt:
                        self._process_simple_statement_line(stmt, None, self.i)
                    self.i = max(self.i, end_idx) + 1

        # Fallback/global parsing to ensure settings exist before tooltips
        self._parse_settings_and_tooltips_global()

        if "ACContinueRooms" not in self.vars:
            idx = self._find_statement_index("vars.ACContinueRooms")
            arr = self._parse_array_literal_from_script(idx, None)
            if arr is not None:
                self.vars["ACContinueRooms"] = arr
        if "OSTRooms" not in self.vars:
            idx = self._find_statement_index("vars.OSTRooms")
            arr = self._parse_array_literal_from_script(idx, None)
            if arr is not None:
                self.vars["OSTRooms"] = arr
        if "firstUpdateDone" not in self.vars:
            self.vars["firstUpdateDone"] = False

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
                m_arr = re.match(r"^new\s+Dictionary[^{\[]*\[(\d+)\]\s*$", value)
                if m_arr:
                    try:
                        size = int(m_arr.group(1))
                    except ValueError:
                        size = 0
                    return [None] * size
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
    asl = ASLInterpreter(asl_script="../asl_scripts/DELTARUNE.asl")
    asl.start_runtime()
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt as e:
        print("Exiting...")
