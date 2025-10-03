# region Imports
import math
import time
from datetime import datetime

import xml.etree.ElementTree as ET
from pathlib import Path

from gi.repository import GObject, GLib, Gio

# endregion


class LSSBaseTime(GObject.Object):
    __gtype__name__ = "LSSBaseTime"

    def __init__(
        self,
        real_time: float = float(1e308),
        game_time: float = float(1e308),
        pause_time: float = float(1e308),
        time_id: str = "0",
        name: str = "",
    ):
        super().__init__()
        self.name = name
        self.time_id = time_id
        self.real_time = real_time
        self.game_time = game_time
        self.pause_time = pause_time


class LSSSegment(GObject.Object):
    __gtype__name__ = "LSSSegment"

    def __init__(
        self,
        name: str,
        icon: str,
        splits: list,
        best_segment_time: LSSBaseTime,
        segment_history: list,
    ):
        super().__init__()
        self.name = name
        self.icon = icon
        self.splits = splits
        self.best_segment_time = best_segment_time
        self.segment_history = segment_history


class LSSMetadata(GObject.Object):
    __gtype_name__ = "LSSMetadata"
    run_id = GObject.Property(type=str, default="")
    platform = GObject.Property(type=str, default="")
    usesEmulator = GObject.Property(type=bool, default=False)
    region = GObject.Property(type=str, default="")
    variables = []

    def __init__(self, xml_tree):
        super().__init__()
        self.parse_metadata(xml_tree)

    def parse_metadata(self, xml_tree):
        for child in xml_tree:
            if child.tag == "Run":
                self.run_id = child.attrib.get("id", "")
            if child.tag == "Platform":
                self.platform = child.text
                self.usesEmulator = (
                    str(child.attrib.get("usesEmulator", "false")).lower() == "true"
                )
            if child.tag == "Region":
                self.region = child.text
            if child.tag == "Variables":
                self.variables = []
                for var in child.findall("Variable"):
                    name = var.attrib.get("name", "")
                    value = var.text if var.text is not None else ""
                    variable = f"{name}: {value}"
                    self.variables.append(variable)


class LSSObject(GObject.Object):
    __gtype_name__ = "LSSObject"
    game_name = GObject.Property(type=str, default="")
    category_name = GObject.Property(type=str, default="")
    attempt_count = GObject.Property(type=int, default=0)
    attempt_history = []
    offset = GObject.Property(type=float, default=0.0)
    segments = []
    if_autosplitter = GObject.Property(type=bool, default=False)
    autosplitter_settings = {}

    def __init__(self, lss_path: str | None = None):
        super().__init__()
        if lss_path is None:
            lss_path = str(Path(__file__).parent.joinpath("example.lss"))
        self.path = lss_path
        self._tree = ET.parse(lss_path)
        self._root = self._tree.getroot()
        self._autosplitter_element: ET.Element | None = None
        self._custom_settings_element: ET.Element | None = None
        root = self._root
        for child in root:
            if child.tag == "GameName":
                self.game_name = child.text
            elif child.tag == "CategoryName":
                self.category_name = child.text
            elif child.tag == "AttemptCount":
                self.attempt_count = int(child.text)
            elif child.tag == "Offset":
                self.offset = self.convert_time_string(child.text)
            elif child.tag == "AttemptHistory":
                self.attempt_history = []
                for attempt in child.findall("Attempt"):
                    real_time = self.convert_time_string(
                        attempt.find("RealTime").text
                        if attempt.find("RealTime") is not None
                        else "00:00:00"
                    )
                    game_time = self.convert_time_string(
                        attempt.find("GameTime").text
                        if attempt.find("GameTime") is not None
                        else "00:00:00"
                    )
                    pause_time = self.convert_time_string(
                        attempt.find("PauseTime").text
                        if attempt.find("PauseTime") is not None
                        else "00:00:00"
                    )
                    self.attempt_history.append(
                        LSSBaseTime(
                            real_time,
                            game_time,
                            pause_time,
                            time_id=attempt.attrib.get("id"),
                        )
                    )
            elif child.tag == "Segments":
                self.segments = self.save_segments(child)
            elif child.tag == "Metadata":
                self.metadata = LSSMetadata(child)
            elif child.tag == "AutoSplitterSettings":
                self.if_autosplitter = True
                self.autosplitter_settings = {}
                self._autosplitter_element = child
                custom_settings = child.find("CustomSettings")
                if custom_settings is None:
                    custom_settings = ET.SubElement(child, "CustomSettings")
                self._custom_settings_element = custom_settings
                for setting in custom_settings.findall("Setting"):
                    if setting.attrib.get("type") == "bool":
                        setting_id = setting.attrib.get("id", "")
                        value = setting.text if setting.text is not None else ""
                        self.autosplitter_settings[setting_id] = value.lower() == "true"

    def set_autosplitter_settings(self, settings: dict[str, bool]):
        if not settings:
            return
        self.if_autosplitter = True
        if self._autosplitter_element is None:
            self._autosplitter_element = ET.SubElement(
                self._root, "AutoSplitterSettings"
            )
        if self._custom_settings_element is None:
            self._custom_settings_element = ET.SubElement(
                self._autosplitter_element, "CustomSettings"
            )

        existing = {
            setting.attrib.get("id", ""): setting
            for setting in self._custom_settings_element.findall("Setting")
        }

        for key, value in settings.items():
            node = existing.get(key)
            if node is None:
                node = ET.SubElement(
                    self._custom_settings_element, "Setting", id=key, type="bool"
                )
            node.set("type", "bool")
            node.set("id", key)
            node.text = "True" if value else "False"

        self.autosplitter_settings = {k: bool(v) for k, v in settings.items()}

    def save(self, path: str | None = None):
        target = path or self.path
        if not target:
            return
        self._tree.write(target, encoding="utf-8", xml_declaration=True)

    def save_segments(self, xml_tree):
        segments = []
        for segment in xml_tree.findall("Segment"):
            name = segment.find("Name").text
            icon = segment.find("Icon").text
            split_times = []
            best_segment_time = LSSBaseTime(
                self.convert_time_string(
                    segment.find("BestSegmentTime").find("RealTime").text
                    if segment.find("BestSegmentTime").find("RealTime") is not None
                    else "00:00:00"
                ),
                self.convert_time_string(
                    segment.find("BestSegmentTime").find("GameTime").text
                    if segment.find("BestSegmentTime").find("GameTime") is not None
                    else "00:00:00"
                ),
            )
            segment_history = []
            for split_time in segment.find("SplitTimes"):
                split_name = split_time.attrib.get("name", "")
                real_time = (
                    self.convert_time_string(split_time.find("RealTime").text)
                    if split_time.find("RealTime") is not None
                    else 0
                )
                game_time = (
                    self.convert_time_string(split_time.find("GameTime").text)
                    if split_time.find("GameTime") is not None
                    else 0
                )
                split_times.append(LSSBaseTime(real_time, game_time, name=split_name))
            for split_time in segment.find("SegmentHistory"):
                real_time = self.convert_time_string(split_time.find("RealTime").text)
                game_time = self.convert_time_string(split_time.find("GameTime").text)
                segment_history.append(
                    LSSBaseTime(
                        real_time, game_time, time_id=split_time.attrib.get("id", "")
                    )
                )
            segments.append(
                LSSSegment(name, icon, split_times, best_segment_time, segment_history)
            )
        return segments

    def convert_time_string(self, time_str: str) -> float:
        if time_str is None or time_str == "":
            return float(1e308)
        try:
            h, m, s = time_str.split(":")
            total_sec = int(h) * 3600 + int(m) * 60 + float(s)
            return total_sec
        except ValueError:
            return float(1e308)


if __name__ == "__main__":
    lss = LSSObject("src/lssparser/example.lss")
    print(lss.game_name)
    print(lss.category_name)
    print(lss.attempt_count)
    print(lss.attempt_history)
    print(lss.offset)
    for segment in lss.segments:
        print(segment.name)
        print(segment.best_segment_time.real_time)
