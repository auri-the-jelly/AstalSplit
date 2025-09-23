import math
import time
from gi.repository import Gio, GObject, Gtk


# ---- Model ----
class SplitItem(GObject.GObject):
    name = GObject.Property(type=str)
    current = GObject.Property(type=str, default="00:00.00")
    best = GObject.Property(type=str, default="00:00.00")
    current_time = GObject.Property(type=float, default=float(1e308))
    best_time = GObject.Property(type=float, default=float(1e308))
    diff = GObject.Property(type=str, default="")

    def __init__(
        self,
        name,
        current="00:00.00",
        best="00:0.00",
        current_time=1e308,
        best_time=1e308,
        diff="",
    ):
        super().__init__(
            name=name,
            current=current,
            best=best,
            current_time=current_time,
            best_time=best_time,
            diff="",
        )


# ---- ListStore wrapper ----
class SplitsList(Gio.ListStore):
    def __init__(self, splits):
        # IMPORTANT: tell ListStore what it contains
        super().__init__(item_type=SplitItem)
        self.update_rows(splits)

    def update_rows(self, splits):
        # rebuild the store (simple approach)
        self.remove_all()
        for s in splits:
            # expected keys: name, best, best_time
            item = SplitItem("Split 1")
            if type(s) == dict:
                item = SplitItem(
                    name=s["name"],
                    current="00:00.00",
                    best=s.get("best", "00:00.00"),
                    current_time=s.get("current_time", float("inf")),
                    best_time=s.get("best_time", float("inf")),
                    diff="",
                )
            else:
                item = s
            self.append(item)


# ---- View ----
class SplitsBox(Gtk.ListBox):
    __gtype_name__ = "SplitsBox"
    end_signal = GObject.Signal("end_signal")

    def __init__(self, splits):
        super().__init__()
        self.set_selection_mode(Gtk.SelectionMode.NONE)
        self.set_css_classes(["SplitsBox"])
        self.set_name("SplitsBox")
        self.update_splits(splits)
        # bind model -> rows
        self.bind_model(self.splits_list, self._create_row)

    def update_splits(self, splits):
        # model
        self.splits_list = splits

    def _create_row(self, item: SplitItem):
        # item is a SplitItem
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        row.set_hexpand(True)

        name_lbl = Gtk.Label(xalign=0, halign=Gtk.Align.START)
        diff_lbl = Gtk.Label(xalign=1, halign=Gtk.Align.END)
        cur_lbl = Gtk.Label(xalign=1, halign=Gtk.Align.END)
        best_lbl = Gtk.Label(xalign=1, halign=Gtk.Align.END)

        # Ensure the times align to the right edge consistently
        # by adding an expanding spacer between name and times.
        spacer = Gtk.Box()
        spacer.set_hexpand(True)

        # Optionally reserve some width for stable columns
        # cur_lbl.set_width_chars(8)
        # best_lbl.set_width_chars(8)
        diff_lbl.set_width_chars(4)

        # initial text
        name_lbl.set_text(
            item.props.name if len(item.props.name) < 10 else item.props.name[:10] + "…"
        )
        diff_lbl.set_text("")
        cur_lbl.set_text(item.props.current)
        best_lbl.set_text(item.props.best)

        # keep labels in sync with model changes
        item.connect(
            "notify::diff", lambda obj, pspec: diff_lbl.set_text(obj.props.diff)
        )
        item.connect(
            "notify::current", lambda obj, pspec: cur_lbl.set_text(obj.props.current)
        )
        item.connect(
            "notify::best", lambda obj, pspec: best_lbl.set_text(obj.props.best)
        )

        # layout
        row.append(name_lbl)
        row.append(spacer)
        row.append(diff_lbl)
        row.append(cur_lbl)
        row.append(best_lbl)
        return row

    def add_split(self, split_time: float, split_string: str):
        # find the first unfilled split (current == "0.00"), then update
        n = self.splits_list.get_n_items()
        for i in range(n):
            item = self.splits_list.get_item(i)  # -> SplitItem
            if item.props.current == "00:00.00":
                item.props.current = split_string
                if split_time < item.props.best_time:
                    item.props.diff = "-" + "{:.2f}".format(
                        item.props.best_time - split_time
                    )
                else:
                    item.props.diff = "+" + "{:.2f}".format(
                        split_time - item.props.best_time
                    )
                if split_time < item.props.best_time:
                    item.props.best_time = split_time
                    item.props.best = split_string
                break
            if i == n - 2:
                self.emit("end_signal")
