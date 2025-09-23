import math
import time
from gi.repository import Astal, GObject, GLib, Gtk, Gio

from timer.ASLSettingsDialog import ASLSettingsDialog
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

        self._settings_dialog = None
        self.timer.connect("settings_requested", self._on_settings_requested)

        self.set_visible(True)

    def _on_settings_requested(self, _timer, interpreter):
        self.show_asl_settings_dialog(interpreter)

    def show_asl_settings_dialog(self, interpreter=None):
        interpreter = interpreter or self.timer.asl_object
        if self._settings_dialog is not None:
            self._settings_dialog.destroy()
            self._settings_dialog = None
        dialog = ASLSettingsDialog(self, self.timer, interpreter)
        dialog.connect("close-request", self._on_settings_dialog_close)
        dialog.present()
        self._settings_dialog = dialog

    def _on_settings_dialog_close(self, dialog):
        self._settings_dialog = None
        return False

    def has_script_settings(self) -> bool:
        return bool(self.timer.script_setting_keys())
