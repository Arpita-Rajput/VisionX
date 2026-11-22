
import os
import json
import re
import time

from datetime import datetime


class RestrictedAreaEventDetector:

    def __init__(
        self,
        zone_x1,
        zone_y1,
        zone_x2,
        zone_y2,
        camera_id=1,
        loitering_threshold=120
    ):

        self.zone_x1 = zone_x1
        self.zone_y1 = zone_y1
        self.zone_x2 = zone_x2
        self.zone_y2 = zone_y2

        self.camera_id = camera_id
        self.loitering_threshold = loitering_threshold

        # Track person states
        self.person_zone_status = {}
        self.entry_times = {}
        self.entry_timestamps = {}
        self.loitering_alerted = set()

        # Default monitoring rules
        self.monitoring_rules = {
            "entry": True,
            "exit": True,
            "loitering": True
        }

        # Event history
        self.output_dir = "output"
        self.event_file = os.path.join(
            self.output_dir,
            "events.json"
        )

        os.makedirs(self.output_dir, exist_ok=True)

        if not os.path.exists(self.event_file):
            with open(
                self.event_file, "w", encoding="utf-8"
            ) as f:
                json.dump([], f, indent=4)

    # ==========================================
    # 1. NATURAL LANGUAGE MONITORING
    # ==========================================

    def configure_from_prompt(self, prompt):

        prompt = prompt.lower()

        # Reset rules for the new prompt
        rules = {
            "entry": False,
            "exit": False,
            "loitering": False
        }

        # Detect entry-related instructions
        if re.search(
            r"\b(enter|enters|entering|entry|intrusion|intrude)\b",
            prompt
        ):
            rules["entry"] = True

        # Detect exit-related instructions
        if re.search(
            r"\b(exit|exits|exiting|leave|leaves|leaving)\b",
            prompt
        ):
            rules["exit"] = True

        # Detect loitering-related instructions
        if re.search(
            r"\b(stay|stays|staying|remain|remains|remaining|"
            r"loiter|loitering|longer|duration)\b",
            prompt
        ):
            rules["loitering"] = True

        # Extract duration from the prompt
        duration_match = re.search(
            r"(\d+(?:\.\d+)?)\s*"
            r"(seconds?|secs?|minutes?|mins?)",
            prompt
        )

        if duration_match:

            value = float(duration_match.group(1))
            unit = duration_match.group(2)

            if unit.startswith(("minute", "min")):
                self.loitering_threshold = int(value * 60)
            else:
                self.loitering_threshold = int(value)

        # If no recognized event was specified,
        # enable all available monitoring features.
        if not any(rules.values()):
            rules = {
                "entry": True,
                "exit": True,
                "loitering": True
            }

        self.monitoring_rules = rules

        print("\n========== MONITORING CONFIGURATION ==========")
        print("User Prompt:", prompt)
        print("Entry Detection:", rules["entry"])
        print("Exit Detection:", rules["exit"])
        print("Loitering Detection:", rules["loitering"])
        print(
            "Loitering Threshold:",
            self.loitering_threshold,
            "seconds"
        )
        print("==============================================\n")

        return rules

    # ==========================================
    # 2. CHECK RESTRICTED AREA
    # ==========================================

    def is_inside_zone(self, bbox):

        x1, y1, x2, y2 = bbox

        # Bounding-box center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        return (
            self.zone_x1 <= center_x <= self.zone_x2
            and self.zone_y1 <= center_y <= self.zone_y2
        )

    # ==========================================
    # 3. EVENT HISTORY
    # ==========================================

    def save_event(self, event):

        with open(
            self.event_file,
            "r",
            encoding="utf-8"
        ) as f:

            events = json.load(f)

        if not isinstance(events, list):
            raise ValueError("events.json must contain a JSON list")

        events.append(event)

        with open(
            self.event_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(events, f, indent=4)

    # ==========================================
    # CREATE AND LOG EVENT
    # ==========================================

    def create_event(
        self,
        track_id,
        event_type,
        message,
        duration_seconds=None,
        entry_timestamp=None
    ):

        current_time = datetime.now()

        event = {
            "camera_id": self.camera_id,
            "track_id": int(track_id),
            "class_name": "person",
            "event": event_type,
            "message": message,
            "timestamp": current_time.isoformat(),
            "duration_seconds": (
                round(duration_seconds, 2)
                if duration_seconds is not None
                else None
            ),
            "entry_timestamp": (
                entry_timestamp.isoformat()
                if entry_timestamp is not None
                else None
            )
        }

        self.save_event(event)

        return event

    # ==========================================
    # 4. PERSON ENTRY DETECTION
    # ==========================================

    def detect_entry(self, track_id):

        current_time = datetime.now()

        event = self.create_event(
            track_id,
            "restricted_area_entry",
            (
                "Person entered the restricted area at "
                f"{current_time.strftime('%I:%M:%S %p')}"
            )
        )

        print("\n========== ENTRY ALERT ==========")
        print(event["message"])
        print("Camera ID:", self.camera_id)
        print("Person ID:", track_id)
        print("=================================\n")

    # ==========================================
    # 5. PERSON EXIT DETECTION
    # ==========================================

    def detect_exit(self, track_id):

        current_time = datetime.now()

        start_time = self.entry_times.get(track_id)
        entry_timestamp = self.entry_timestamps.get(track_id)

        duration = (
            time.monotonic() - start_time
            if start_time is not None
            else None
        )

        event = self.create_event(
            track_id,
            "restricted_area_exit",
            (
                "Person exited the restricted area at "
                f"{current_time.strftime('%I:%M:%S %p')}"
            ),
            duration_seconds=duration,
            entry_timestamp=entry_timestamp
        )

        print("\n========== EXIT ALERT ==========")
        print(event["message"])
        print("Camera ID:", self.camera_id)
        print("Person ID:", track_id)

        if duration is not None:
            print("Time spent:", round(duration, 2), "seconds")

        print("================================\n")

    # ==========================================
    # 6. LOITERING DETECTION
    # ==========================================

    def detect_loitering(self, track_id):

        if track_id not in self.entry_times:
            return

        duration = (
            time.monotonic() - self.entry_times[track_id]
        )

        if (
            duration >= self.loitering_threshold
            and track_id not in self.loitering_alerted
        ):

            current_time = datetime.now()

            event = self.create_event(
                track_id,
                "loitering_alert",
                (
                    f"Person has stayed in the restricted area "
                    f"for {round(duration, 2)} seconds"
                ),
                duration_seconds=duration,
                entry_timestamp=self.entry_timestamps.get(track_id)
            )

            print("\n========== LOITERING ALERT ==========")
            print(event["message"])
            print("Camera ID:", self.camera_id)
            print("Person ID:", track_id)
            print("=====================================\n")

            self.loitering_alerted.add(track_id)

    # ==========================================
    # 7. PROCESS ALL DETECTIONS
    # ==========================================

    def process_detections(self, detections):

        new_events = []

        for detection in detections:

            if detection["class_name"] != "person":
                continue

            track_id = detection["track_id"]

            if track_id is None:
                continue

            track_id = int(track_id)

            bbox = detection["bbox"]

            current_inside = self.is_inside_zone(bbox)

            previous_inside = self.person_zone_status.get(
                track_id, False
            )

            # ----------------------------------
            # PERSON ENTERED
            # ----------------------------------

            if current_inside and not previous_inside:

                self.entry_times[track_id] = time.monotonic()
                self.entry_timestamps[track_id] = datetime.now()

                self.loitering_alerted.discard(track_id)

                if self.monitoring_rules["entry"]:
                    self.detect_entry(track_id)

            # ----------------------------------
            # PERSON INSIDE: CHECK LOITERING
            # ----------------------------------

            if current_inside:

                if self.monitoring_rules["loitering"]:
                    self.detect_loitering(track_id)

            # ----------------------------------
            # PERSON EXITED
            # ----------------------------------

            elif previous_inside:

                if self.monitoring_rules["exit"]:
                    self.detect_exit(track_id)

                self.entry_times.pop(track_id, None)
                self.entry_timestamps.pop(track_id, None)
                self.loitering_alerted.discard(track_id)

            # Update tracking state
            self.person_zone_status[track_id] = current_inside

        return new_events

    # ==========================================
    # 8. EVENT HISTORY SUMMARY
    # ==========================================

    def get_event_history(self):

        with open(
            self.event_file,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    def get_event_summary(self):

        events = self.get_event_history()

        summary = {
            "total_events": len(events),
            "entries": 0,
            "exits": 0,
            "loitering_alerts": 0
        }

        for event in events:

            if event["event"] == "restricted_area_entry":
                summary["entries"] += 1

            elif event["event"] == "restricted_area_exit":
                summary["exits"] += 1

            elif event["event"] == "loitering_alert":
                summary["loitering_alerts"] += 1

        return summary
