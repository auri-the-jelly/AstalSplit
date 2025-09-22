import math
import time
from gi.repository import Astal, GObject, GLib, Gtk, GObject, Gio

from timer.TimerBox import TimerBox
from timer.Timer import Timer


class TimerWindow(Astal.Window):
    __gtype_name__ = "TimerWindow"

    def __init__(self):
        super().__init__(
            anchor=Astal.WindowAnchor.TOP | Astal.WindowAnchor.RIGHT,
            exclusivity=Astal.Exclusivity.IGNORE,
            css_classes=["Timer"],
            visible=False,
            layer=Astal.Layer.OVERLAY,
            margin_top=24,
        )
        self.timer = Timer()
        self.timer_box = TimerBox(self.timer)
        self.set_child(self.timer_box)
        self.layer = Astal.Layer.OVERLAY

        self.set_visible(True)
