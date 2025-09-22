import math
import time
from gi.repository import Astal, GObject, GLib, Gtk, GObject, Gio

from lssparser.LSSParse import LSSObject
from timer.SplitsBox import SplitItem, SplitsList
from aslrunner.asl_parser import ASLSettings, ASLSetting, ASLInterpreter

SYNC = GObject.BindingFlags.SYNC_CREATE

splits = [
    {
        "name": "Split 1",
        "current": "00:00.00",
        "best": "00:00.00",
        "current_time": float(1e308),
        "best_time": float(1e308),
    },
]


class Timer(GObject.Object):
    __gtype_name__ = "Timer"
    time_s = 0.0
    time_string = GObject.Property(type=str, default="00:00.00")
    running = GObject.Property(type=bool, default=False)
    segments = SplitsList(splits=splits)
    split_signal = GObject.Signal("split_signal")
    reset_signal = GObject.Signal("reset_signal")
    start_signal = GObject.Signal("start_signal")
    pause_signal = GObject.Signal("pause_signal")

    def __init__(self):
        super().__init__()
        self.t0 = 0.0
        self.accum = 0.0
        self.on_interval()
        self.splits = splits
        self.cur_splits = []
        self.asl_object = None
        GLib.timeout_add(25, self.on_interval, GLib.PRIORITY_HIGH)

    def on_interval(self, *_args):
        if self.running:
            self.time_s = self.current_elapsed()
            self.time_string = self.format_time(self.time_s)
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
        if self.t0 != 0.0:
            t = self.current_elapsed()
            self.cur_splits.append(t)
            self.emit("split_signal")

    def on_reset(self):
        self.running = False
        self.t0 = 0.0
        self.accum = 0.0
        self.time_string = "00:00.00"
        self.splits.clear()
        self.emit("reset_signal")

    def load_splits(self, lss_path: str):
        self.splits.clear()
        tmp_segments = []
        lss = LSSObject(lss_path)
        if lss.if_autosplitter:
            self.asl_object = ASLInterpreter(
                game_name=lss.game_name, asl_settings=lss.autosplitter_settings
            )
            self.asl_object.connect("split_signal", lambda *_: self.on_split())
            self.asl_object.connect("reset_signal", lambda *_: self.on_reset())
            self.asl_object.connect(
                "start_signal", lambda *_: self.on_start_pause(True)
            )
            self.asl_object.connect(
                "pause_signal", lambda *_: self.on_start_pause(False)
            )
            GLib.timeout_add(50.0 / 3.0, self.asl_object.state_update)

        for segment in lss.segments:
            split_item = SplitItem(
                segment.name,
                "00:00.00",
                self.format_time(segment.best_segment_time.game_time),
                float(1e308),
                segment.best_segment_time.game_time,
            )
            tmp_segments.append(split_item)
        self.segments.update_rows(tmp_segments)

    def current_elapsed(self) -> float:
        if self.running:
            return self.accum + (time.perf_counter() - self.t0)
        return self.accum

    def format_time(self, sec: float) -> str:
        # mm:ss.mmm (minutes:seconds.milliseconds)
        m, s = divmod(sec, 60.0)
        m = int(m)
        return f"{m:02d}:{s:05.2f}"
