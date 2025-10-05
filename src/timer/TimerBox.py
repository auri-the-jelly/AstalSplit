import math
import time
from gi.repository import Astal, GObject, GLib, Gtk, GObject, Gio

from timer.SplitsBox import SplitsBox, SplitsList

SYNC = GObject.BindingFlags.SYNC_CREATE


class TimerBox(Gtk.Box):
    __gtype_name__ = "TimerBox"
    timer_string = GObject.Property(type=str, default="00:00.00")
    timer = None

    def __init__(self, Timer):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.timer = Timer
        self.splits_box = SplitsBox(self.timer.segments)
        self.time_label = Gtk.Label(label=Timer.time_string)
        self.timer.connect("split_signal", self.on_split)
        self.timer.bind_property("time_string", self.time_label, "label", SYNC)
        self.append(self.time_label)
        self.append(self.splits_box)

    def on_split(self, *_):
        self.splits_box.add_split(
            self.timer.cur_splits[-1], self.timer.format_time(self.timer.cur_splits[-1])
        )
