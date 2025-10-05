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
        self._attempt_history_element: ET.Element | None = None
        self._attempt_count_element: ET.Element | None = None
        self._segments_element: ET.Element | None = None

        root = self._root
        for child in root:
            if child.tag == "GameName":
                self.game_name = child.text
            elif child.tag == "CategoryName":
                self.category_name = child.text
            elif child.tag == "AttemptCount":
                self._attempt_count_element = child
                self.attempt_count = int(child.text)
            elif child.tag == "Offset":
                self.offset = self.convert_time_string(child.text)
            elif child.tag == "AttemptHistory":
                self._attempt_history_element = child
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
                self._segments_element = child
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

    @staticmethod
    def _format_seconds(seconds: float) -> str | None:
        """Convert seconds to LiveSplit's hh:mm:ss.fffffff format."""

        if seconds is None or not math.isfinite(seconds) or seconds >= float(1e308):
            return None
        total_ticks = int(round(seconds * 10_000_000))
        if total_ticks < 0:
            total_ticks = 0
        seconds_part, fractional = divmod(total_ticks, 10_000_000)
        minutes, sec = divmod(seconds_part, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{sec:02d}.{fractional:07d}"

    def _next_attempt_id(self) -> str:
        """Compute the next attempt id, preserving numeric ordering when possible."""

        existing_ids: list[int] = []
        if self._attempt_history_element is not None:
            for attempt in self._attempt_history_element.findall("Attempt"):
                try:
                    existing_ids.append(int(attempt.attrib.get("id", "0")))
                except ValueError:
                    continue
        if existing_ids:
            return str(max(existing_ids) + 1)
        return str(self.attempt_count + 1 if self.attempt_count else 1)

    def add_completed_run(
        self,
        cumulative_real_times: list[float],
        cumulative_game_times: list[float] | None = None,
    ) -> bool:
        """Append a completed run to the LSS data structures.

        Args:
            cumulative_real_times: cumulative real-time splits, in seconds.
            cumulative_game_times: optional cumulative game-time splits.

        Returns:
            True if the run was recorded, False otherwise.
        """

        if not cumulative_real_times:
            return False
        if self._segments_element is None:
            return False
        segments_xml = self._segments_element.findall("Segment")
        if not segments_xml:
            return False

        segment_count = min(len(segments_xml), len(cumulative_real_times))
        if segment_count == 0:
            return False

        now = datetime.now()
        timestamp = now.strftime("%m/%d/%Y %H:%M:%S")
        attempt_id = self._next_attempt_id()

        # -- Update attempt count --
        self.attempt_count += 1
        if self._attempt_count_element is not None:
            self._attempt_count_element.text = str(self.attempt_count)

        # -- Append to AttemptHistory --
        if self._attempt_history_element is not None:
            attempt_el = ET.SubElement(
                self._attempt_history_element,
                "Attempt",
                id=attempt_id,
                started=timestamp,
                isStartedSynced="True",
                ended=timestamp,
                isEndedSynced="True",
            )
            total_real = cumulative_real_times[segment_count - 1]
            real_text = self._format_seconds(total_real)
            if real_text:
                ET.SubElement(attempt_el, "RealTime").text = real_text
            game_times_source = cumulative_game_times or cumulative_real_times
            if game_times_source:
                total_game = game_times_source[segment_count - 1]
                game_text = self._format_seconds(total_game)
                if game_text:
                    ET.SubElement(attempt_el, "GameTime").text = game_text
            self.attempt_history.append(
                LSSBaseTime(
                    total_real,
                    game_times_source[segment_count - 1]
                    if game_times_source
                    else float(1e308),
                )
            )

        # -- Update segment metadata --
        previous_real = 0.0
        previous_game = 0.0
        game_times_source = cumulative_game_times or cumulative_real_times

        for index in range(segment_count):
            segment_el = segments_xml[index]
            seg_real_cumulative = cumulative_real_times[index]
            seg_real_delta = seg_real_cumulative - previous_real
            previous_real = seg_real_cumulative

            seg_game_delta = float(1e308)
            if game_times_source:
                seg_game_cumulative = game_times_source[index]
                seg_game_delta = seg_game_cumulative - previous_game
                previous_game = seg_game_cumulative

            # Update SegmentHistory
            segment_history_el = segment_el.find("SegmentHistory")
            if segment_history_el is None:
                segment_history_el = ET.SubElement(segment_el, "SegmentHistory")
            time_el = ET.SubElement(segment_history_el, "Time", id=attempt_id)
            real_history_text = self._format_seconds(seg_real_delta)
            if real_history_text:
                ET.SubElement(time_el, "RealTime").text = real_history_text
            game_history_text = self._format_seconds(seg_game_delta)
            if game_history_text:
                ET.SubElement(time_el, "GameTime").text = game_history_text

            # Update BestSegmentTime when applicable
            best_seg_el = segment_el.find("BestSegmentTime")
            if best_seg_el is None:
                best_seg_el = ET.SubElement(segment_el, "BestSegmentTime")
            best_real_el = best_seg_el.find("RealTime")
            if best_real_el is None:
                best_real_el = ET.SubElement(best_seg_el, "RealTime")
            current_best_real = self.convert_time_string(best_real_el.text)
            if seg_real_delta < current_best_real:
                best_real_text = self._format_seconds(seg_real_delta)
                if best_real_text:
                    best_real_el.text = best_real_text
            best_game_el = best_seg_el.find("GameTime")
            if best_game_el is None:
                best_game_el = ET.SubElement(best_seg_el, "GameTime")
            current_best_game = self.convert_time_string(best_game_el.text)
            if seg_game_delta < current_best_game:
                best_game_text = self._format_seconds(seg_game_delta)
                if best_game_text:
                    best_game_el.text = best_game_text

            # Update Personal Best cumulative time if improved
            split_times_el = segment_el.find("SplitTimes")
            if split_times_el is None:
                split_times_el = ET.SubElement(segment_el, "SplitTimes")
            personal_best_el = None
            for candidate in split_times_el.findall("SplitTime"):
                if candidate.attrib.get("name") == "Personal Best":
                    personal_best_el = candidate
                    break
            if personal_best_el is None:
                personal_best_el = ET.SubElement(
                    split_times_el, "SplitTime", name="Personal Best"
                )
            pb_real_el = personal_best_el.find("RealTime")
            if pb_real_el is None:
                pb_real_el = ET.SubElement(personal_best_el, "RealTime")
            pb_current_real = self.convert_time_string(pb_real_el.text)
            if seg_real_cumulative < pb_current_real:
                pb_real_text = self._format_seconds(seg_real_cumulative)
                if pb_real_text:
                    pb_real_el.text = pb_real_text
            pb_game_el = personal_best_el.find("GameTime")
            if pb_game_el is None:
                pb_game_el = ET.SubElement(personal_best_el, "GameTime")
            pb_current_game = self.convert_time_string(pb_game_el.text)
            if game_times_source:
                seg_game_cumulative = game_times_source[index]
                if seg_game_cumulative < pb_current_game:
                    pb_game_text = self._format_seconds(seg_game_cumulative)
                    if pb_game_text:
                        pb_game_el.text = pb_game_text

            # Keep in-memory representation loosely in sync
            if index < len(self.segments):
                segment_obj = self.segments[index]
                segment_obj.segment_history.append(
                    LSSBaseTime(seg_real_delta, seg_game_delta, time_id=attempt_id)
                )
                if seg_real_delta < segment_obj.best_segment_time.real_time:
                    segment_obj.best_segment_time.real_time = seg_real_delta
                if seg_game_delta < segment_obj.best_segment_time.game_time:
                    segment_obj.best_segment_time.game_time = seg_game_delta
                if segment_obj.splits:
                    for split in segment_obj.splits:
                        if split.name == "Personal Best" and (
                            seg_real_cumulative < split.real_time
                        ):
                            split.real_time = seg_real_cumulative
                            split.game_time = (
                                game_times_source[index]
                                if game_times_source
                                else split.game_time
                            )

        return True

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
