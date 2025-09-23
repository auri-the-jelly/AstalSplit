import os
from pathlib import Path

from gi.repository import Gio, GLib, Gtk, Gdk
from timer.TimerWindow import TimerWindow


def _get_data_root() -> Path:
    env_path = os.environ.get("ASTALSPLIT_DATADIR")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent.parent


gresource = Gio.Resource.load(str(_get_data_root() / "resources.gresource"))
gresource._register()


class App(Gtk.Application):
    __gtype_name__ = "App"

    def _init_css(self):
        provider = Gtk.CssProvider()
        provider.load_from_resource("/style.css")

        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_USER,
        )

    # this is the method that will be invoked on `app.run()`
    # this is where everything should be initialized and instantiated
    def do_command_line(self, command_line):
        argv = command_line.get_arguments()

        if command_line.get_is_remote():
            # app is already running we can print to remote
            if len(argv) > 1 and self.timer_window:
                match (argv[1]):
                    case "start":
                        command_line.print_literal("Started")
                        self.timer_window.timer.on_start_pause(True)
                    case "pause":
                        command_line.print_literal("Paused")
                        self.timer_window.timer.on_start_pause(False)
                    case "toggle":
                        command_line.print_literal("Toggled")
                        self.timer_window.timer.on_start_pause()
                    case "split":
                        command_line.print_literal("Split")
                        self.timer_window.timer.on_split()
                    case "reset":
                        command_line.print_literal("Reset")
                        self.timer_window.timer.on_reset()
                    case "load":
                        if len(argv) > 2:
                            path = ""
                            if os.path.isabs(argv[2]):
                                path = argv[2]
                            else:
                                path = os.path.abspath(
                                    os.path.join(command_line.get_cwd(), argv[2])
                                )
                            self.lss_path = path
                            command_line.print_literal(f"Loaded {self.lss_path}")
                            self.timer_window.timer.load_splits(self.lss_path)
                            self._update_asl_settings_action()
                            return
                        else:
                            command_line.print_literal("usage: astalsplit load <path>")
                    case _:
                        command_line.print_literal(
                            "usage: astalsplit <command> \ncommands:\n    start: start the timer\n    pause: pause the timer\n    toggle: toggle start/pause\n    split: record a split\n    reset: reset the timer\n"
                        )
                return 0
            else:
                command_line.print_literal(
                    "usage: astalsplit <command> \n commands:\n    start: start the timer\n    pause: pause the timer\n    toggle: toggle start/pause\n    split: record a split\n    reset: reset the timer\n"
                )
        else:
            if len(argv) > 2:
                if argv[1] == "load" and os.path.exists(argv[2]):
                    self.lss_path = os.path.abspath(argv[2])
                elif argv[1] == "load" and not os.path.exists(
                    os.path.join(os.getcwd(), argv[2])
                ):
                    print(os.path.join(os.getcwd(), argv[2]))
                    print("Couldn't find file. Ignoring.")
            # main instance, initialize stuff here
            self._init_css()
            self.timer_window = TimerWindow()
            self._make_actions(self.timer_window)
            menu = Gio.Menu()
            menu.append("Start / Pause", "app.startpause")
            menu.append("Split", "app.split")
            menu.append("Reset", "app.reset")
            menu.append("Autosplitter Settings", "app.aslsettings")
            menu.append("Quit", "app.quit")
            self.pop = Gtk.PopoverMenu()
            self.pop.set_has_arrow(False)
            self.pop.set_menu_model(menu)
            self.pop.set_parent(self.timer_window)
            click = Gtk.GestureClick.new()
            click.set_button(0)  # listen to any button; we’ll filter

            def on_pressed(gesture, n_press, x, y):
                b = gesture.get_current_button()
                if b == Gdk.BUTTON_SECONDARY:  # right click
                    # Position the popover under the pointer
                    self.pop.set_pointing_to(Gdk.Rectangle(int(x), int(y), 1, 1))
                    self.pop.popup()

            click.connect("pressed", on_pressed)
            self.timer_window.add_controller(click)
            self.timer_window.timer.load_splits(getattr(self, "lss_path", None))
            self._update_asl_settings_action()
            self.add_window(self.timer_window)

            def on_start_pause(start: bool):
                self.action_split.set_enabled(start)
                self.action_reset.set_enabled(True)

            def on_reset():
                self.action_split.set_enabled(False)
                self.action_reset.set_enabled(False)

            self.timer_window.timer.connect(
                "start_signal", lambda *_: on_start_pause(True)
            )
            self.timer_window.timer.connect(
                "pause_signal", lambda *_: on_start_pause(False)
            )
            self.timer_window.timer.connect("reset_signal", lambda *_: on_reset())

        return 0

    def _make_actions(self, timer_window):
        def add(name, cb, enabled=True):
            a = Gio.SimpleAction.new(name, None)
            a.connect("activate", lambda *_: cb())
            a.set_enabled(enabled)
            self.add_action(a)
            setattr(self, f"action_{name}", a)

        add("startpause", timer_window.timer.on_start_pause)
        add("split", timer_window.timer.on_split, enabled=False)
        add("reset", timer_window.timer.on_reset, enabled=False)
        add(
            "aslsettings",
            lambda: timer_window.show_asl_settings_dialog(),
            enabled=False,
        )
        add("quit", self.quit)

    def _update_asl_settings_action(self):
        if not hasattr(self, "action_aslsettings"):
            return
        has_settings = (
            hasattr(self, "timer_window")
            and self.timer_window is not None
            and self.timer_window.has_script_settings()
        )
        self.action_aslsettings.set_enabled(has_settings)

    def __init__(self) -> None:
        super().__init__(
            application_id="sh.astal.split",
            flags=Gio.ApplicationFlags.HANDLES_COMMAND_LINE,
        )

    @staticmethod
    def main(argv):
        App.instance = App()
        GLib.set_prgname("astalsplit")
        return App.instance.run(argv)
