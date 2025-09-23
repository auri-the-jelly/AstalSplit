from __future__ import annotations

from typing import TYPE_CHECKING

from gi.repository import Gtk, Pango

if TYPE_CHECKING:
    from timer.Timer import Timer


class ASLSettingsDialog(Gtk.Window):
    """Simple hierarchical checkbox dialog for autosplitter settings."""

    def __init__(self, parent: Gtk.Window | None, timer: "Timer", interpreter):
        super().__init__(title="Autosplitter Settings")
        if parent is not None:
            self.set_transient_for(parent)
        self.set_modal(True)
        self.set_destroy_with_parent(True)
        self.set_default_size(420, 480)

        self._timer = timer
        self._interpreter = interpreter
        self._settings = getattr(interpreter, "settings", None)
        self._tree_model = None
        self._widget_cache: dict[str, dict[str, Gtk.Widget]] = {}
        self._setting_objects: dict[str, object] = {}
        self._programmatic_update = 0

        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        outer.set_margin_top(16)
        outer.set_margin_bottom(16)
        outer.set_margin_start(20)
        outer.set_margin_end(20)

        if not self._settings or not getattr(self._settings, "settings", {}):
            outer.append(Gtk.Label(label="No autosplitter settings available."))
            close_btn = Gtk.Button(label="Close")
            close_btn.set_halign(Gtk.Align.END)
            close_btn.connect("clicked", lambda *_: self.close())
            outer.append(close_btn)
            self.set_child(outer)
            return

        self._tree_model = self._settings.export_list_object(passthrough=False)
        selection = Gtk.SingleSelection(model=self._tree_model)

        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_setup)
        factory.connect("bind", self._on_bind)
        factory.connect("unbind", self._on_unbind)

        list_view = Gtk.ListView(model=selection, factory=factory)
        list_view.set_vexpand(True)
        list_view.set_show_separators(False)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_child(list_view)
        outer.append(scrolled)

        button_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        button_row.set_halign(Gtk.Align.END)
        close_btn = Gtk.Button(label="Close")
        close_btn.connect("clicked", lambda *_: self.close())
        button_row.append(close_btn)
        outer.append(button_row)

        self.set_child(outer)

    # region Gtk factory callbacks
    def _on_setup(self, _factory, list_item: Gtk.ListItem):
        expander = Gtk.TreeExpander()
        expander.set_margin_top(4)
        expander.set_margin_bottom(4)

        row_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        row_box.set_hexpand(True)

        toggle = Gtk.CheckButton()
        toggle.set_valign(Gtk.Align.CENTER)

        label = Gtk.Label(xalign=0)
        label.set_hexpand(True)
        label.set_ellipsize(Pango.EllipsizeMode.END)

        row_box.append(toggle)
        row_box.append(label)
        expander.set_child(row_box)

        list_item._expander = expander  # type: ignore[attr-defined]
        list_item._toggle = toggle  # type: ignore[attr-defined]
        list_item._label = label  # type: ignore[attr-defined]
        list_item._toggle_handler = None  # type: ignore[attr-defined]
        list_item._notify_handler = None  # type: ignore[attr-defined]

        list_item.set_child(expander)

    def _on_bind(self, _factory, list_item: Gtk.ListItem):
        row = list_item.get_item()
        if not isinstance(row, Gtk.TreeListRow):
            return

        setting = row.get_item()
        if not setting:
            return

        self._setting_objects[setting.setting_name] = setting

        expander = list_item._expander  # type: ignore[attr-defined]
        toggle = list_item._toggle  # type: ignore[attr-defined]
        label = list_item._label  # type: ignore[attr-defined]

        expander.set_list_row(row)

        label.set_text(setting.display_name or setting.setting_name)
        tooltip = setting.tooltip or ""
        label.set_tooltip_text(tooltip or None)
        toggle.set_tooltip_text(tooltip or None)

        meta = self._settings.settings.get(setting.setting_name) if self._settings else None
        if meta is not None:
            desired = bool(meta.get("value", False))
            if bool(setting.setting_value) != desired:
                setting.setting_value = desired

        toggle.set_active(bool(setting.setting_value))

        # Keep checkbox in sync if value changes elsewhere.
        notify_id = setting.connect(
            "notify::setting_value",
            lambda obj, _pspec: toggle.set_active(bool(obj.setting_value)),
        )
        list_item._notify_handler = (setting, notify_id)  # type: ignore[attr-defined]

        def on_toggled(button, gobj):
            if self._programmatic_update > 0:
                return
            new_val = bool(button.get_active())
            if gobj.setting_value != new_val:
                gobj.setting_value = new_val
            meta = self._settings.settings.get(gobj.setting_name)
            if meta is not None:
                meta["value"] = new_val
            self._refresh_row_state(gobj.setting_name)
            self._update_descendants(gobj.setting_name)
            self._persist_settings()

        handler_id = toggle.connect("toggled", on_toggled, setting)
        list_item._toggle_handler = (toggle, handler_id)  # type: ignore[attr-defined]
        self._widget_cache[setting.setting_name] = {
            "toggle": toggle,
            "label": label,
        }

        self._refresh_row_state(setting.setting_name)

    def _on_unbind(self, _factory, list_item: Gtk.ListItem):
        handler = getattr(list_item, "_toggle_handler", None)
        if handler is not None:
            toggle, handler_id = handler
            toggle.disconnect(handler_id)
            list_item._toggle_handler = None

        notify = getattr(list_item, "_notify_handler", None)
        if notify is not None:
            setting, notify_id = notify
            setting.disconnect(notify_id)
            list_item._notify_handler = None

        row = list_item.get_item()
        if isinstance(row, Gtk.TreeListRow):
            setting = row.get_item()
            if setting:
                self._widget_cache.pop(setting.setting_name, None)
                self._setting_objects.pop(setting.setting_name, None)

        expander = getattr(list_item, "_expander", None)
        if expander is not None:
            expander.set_list_row(None)

    # endregion

    def _refresh_row_state(self, name: str):
        widgets = self._widget_cache.get(name)
        if not widgets:
            return
        toggle = widgets.get("toggle")
        label = widgets.get("label")
        is_enabled = self._parents_enabled(name)
        if toggle is not None:
            toggle.set_sensitive(is_enabled)
        if label is not None:
            label.set_sensitive(is_enabled)

    def _parents_enabled(self, name: str) -> bool:
        if not self._settings:
            return True
        meta = self._settings.settings.get(name)
        parent_name = (meta.get("parent") or "") if meta else ""
        while parent_name:
            parent_meta = self._settings.settings.get(parent_name)
            if not parent_meta or not parent_meta.get("value", False):
                return False
            parent_name = (parent_meta.get("parent") or "") if parent_meta else ""
        return True

    def _update_descendants(self, parent_name: str):
        if not self._settings:
            return
        for child_name, meta in self._settings.settings.items():
            if (meta.get("parent") or "") == parent_name:
                if not self._parents_enabled(child_name):
                    self._set_setting_value(child_name, False)
                self._refresh_row_state(child_name)
                self._update_descendants(child_name)

    def _set_setting_value(self, name: str, value: bool):
        if not self._settings:
            return
        meta = self._settings.settings.get(name)
        if meta is not None:
            meta["value"] = value
        setting = self._setting_objects.get(name)
        if setting and getattr(setting, "setting_value", None) == value:
            return
        self._programmatic_update += 1
        try:
            if setting and getattr(setting, "setting_value", None) != value:
                setting.setting_value = value
        finally:
            self._programmatic_update -= 1

    def _persist_settings(self):
        if self._timer:
            try:
                self._timer.persist_asl_settings()
            except Exception as exc:
                print(f"Failed to persist autosplitter settings: {exc}")
