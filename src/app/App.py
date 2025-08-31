from gi.repository import Gio, GLib, Gtk, Gdk, GLib
from timer.TimerWindow import TimerWindow

gresource = Gio.Resource.load("resources.gresource")
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
                return
        else:
            # main instance, initialize stuff here
            self._init_css()
            self.timer_window = TimerWindow()
            self._make_actions(self.timer_window)
            menu = Gio.Menu()
            menu.append("Start / Pause", "app.startpause")
            menu.append("Split", "app.split")
            menu.append("Reset", "app.reset")
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
        add("quit", self.quit)

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
