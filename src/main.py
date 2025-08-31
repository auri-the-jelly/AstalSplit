from ctypes import CDLL

CDLL("libgtk4-layer-shell.so")

import time
from sys import argv, path
import gi

gi.require_version("Gio", "2.0")
gi.require_version("GObject", "2.0")
gi.require_version("GLib", "2.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Astal", "4.0")

if __name__ == "__main__":
    from app.App import App

    App.main(argv)
