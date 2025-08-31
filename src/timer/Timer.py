import math
import time
from gi.repository import Astal, GObject, GLib, Gtk, GObject, Gio

SYNC = GObject.BindingFlags.SYNC_CREATE


class Timer(GObject.Object):
    __gtype_name__ = "Timer"
    time_ms = 0.0
    time_string = GObject.Property(type=str, default="0:00.000")
    running = GObject.Property(type=bool, default=False)
    splits = []

    split_signal = GObject.Signal("split_signal")
    reset_signal = GObject.Signal("reset_signal")
    start_signal = GObject.Signal("start_signal")
    pause_signal = GObject.Signal("pause_signal")

    def __init__(self):
        super().__init__()
        self.t0 = 0.0
        self.accum = 0.0
        self.on_interval()
        GLib.timeout_add(25, self.on_interval, GLib.PRIORITY_HIGH)

    def on_interval(self, *_args):
        if self.running:
            self.time_ms = self.current_elapsed()
            self.time_string = self.format_time(self.time_ms)
        return GLib.SOURCE_CONTINUE

    def on_start_pause(self, if_start=None, *_):
        if if_start is None:
            if not self.running:
                self.start()
            elif self.running:
                self.pause()
        elif if_start:
            self.start()
        elif not if_start:
            self.pause()

    def pause(self):
        self.accum = self.current_elapsed()
        self.running = False
        self.emit("pause_signal")

    def start(self):
        self.t0 = time.perf_counter()
        self.running = True
        self.emit("start_signal")

    def on_split(self):
        t = self.current_elapsed()
        self.splits.append(t)
        self.emit("split_signal")

    def on_reset(self):
        self.running = False
        self.t0 = 0.0
        self.accum = 0.0
        self.time_string = "00:00.00"
        self.splits.clear()
        self.emit("reset_signal")

    def current_elapsed(self) -> float:
        if self.running:
            return self.accum + (time.perf_counter() - self.t0)
        return self.accum

    def format_time(self, sec: float) -> str:
        # mm:ss.mmm (minutes:seconds.milliseconds)
        m, s = divmod(sec, 60.0)
        m = int(m)
        return f"{m:02d}:{s:05.2f}"
