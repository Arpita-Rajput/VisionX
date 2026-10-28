
import cv2
import time
import json
import os

from datetime import datetime
from camera.camera_manager import CameraManager
from detection.yolo_detector import YOLODetector


def main():

    # Create camera object
    camera = CameraManager(0)

    # Load YOLO model
    detector = YOLODetector()

    # Initialize frame counter
    frame_count = 0

    # People counting settings
    line_y = 300
    people_entered = 0
    previous_positions = {}
    counted_ids = set()

    # Restricted area coordinates
    zone_x1 = 200
    zone_y1 = 100
    zone_x2 = 500
    zone_y2 = 400

    # Restricted area tracking
    inside_zone_ids = set()
    zone_entry_times = {}
    alerted_ids = set()

    # --------------------------------
    # EVENT LOGGING SETUP
    # --------------------------------

    event_file = "output/events.json"

    os.makedirs("output", exist_ok=True)

    if not os.path.exists(event_file):
        with open(event_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)

    print("Camera started...")
    print("Press 'q' to quit.")

    try:
        while True:

            # Read frame from camera
            frame = camera.read_frame()

            if frame is None:
                print("Unable to read frame from camera.")
                break

            frame_count += 1

            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]

            # Adjust counting line to fit the frame
            current_line_y = min(
                max(line_y, 0),
                frame_height - 1
            )

            # Adjust restricted zone to fit the frame
            zx1 = max(0, min(zone_x1, frame_width - 1))
            zy1 = max(0, min(zone_y1, frame_height - 1))
            zx2 = max(0, min(zone_x2, frame_width - 1))
            zy2 = max(0, min(zone_y2, frame_height - 1))

            # Detect and track objects
            results = detector.track_objects(frame)

            # Extract tracking information
            detections = detector.extract_tracking_data(results)

            # IDs detected in the current frame
            detected_person_ids = set()

            # Process tracked people
            for detection in detections:

                # Consider only people
                if detection["class_name"] != "person":
                    continue

                track_id = detection["track_id"]
                x1, y1, x2, y2 = detection["bbox"]

                detected_person_ids.add(track_id)

                # Bottom-center point of the person
                center_x = int((x1 + x2) / 2)
                bottom_y = int(y2)

                # --------------------------------
                # 1. PEOPLE COUNTING
                # --------------------------------

                if track_id in previous_positions:

                    previous_y = previous_positions[track_id]

                    # Count top-to-bottom crossings once per ID
                    if (
                        previous_y < current_line_y
                        and bottom_y >= current_line_y
                        and track_id not in counted_ids
                    ):
                        people_entered += 1
                        counted_ids.add(track_id)

                        print(
                            f"Person ID {track_id} crossed the line. "
                            f"Total entered: {people_entered}"
                        )

                previous_positions[track_id] = bottom_y

                # --------------------------------
                # 2. RESTRICTED AREA DETECTION
                # --------------------------------

                inside_zone = (
                    zx1 <= center_x <= zx2
                    and zy1 <= bottom_y <= zy2
                )

                if inside_zone:

                    # Detect first entry into the zone
                    if track_id not in inside_zone_ids:

                        print(
                            f"ALERT: Person {track_id} "
                            "entered restricted area!"
                        )

                        inside_zone_ids.add(track_id)
                        zone_entry_times[track_id] = time.time()

                    # Calculate time spent in the zone
                    duration = (
                        time.time() - zone_entry_times[track_id]
                    )

                    # Alert once after 10 seconds
                    if (
                        duration >= 10
                        and track_id not in alerted_ids
                    ):

                        print(
                            f"ALERT: Person {track_id} stayed "
                            "in the restricted area for 10 seconds!"
                        )

                        # Save event to JSON
                        event = {
                            "event": "Restricted zone dwell-time alert",
                            "track_id": int(track_id),
                            "timestamp": datetime.now().isoformat(),
                            "duration_seconds": round(duration, 2)
                        }

                        # Read existing events
                        with open(
                            event_file, "r", encoding="utf-8"
                        ) as f:
                            events = json.load(f)

                        # Add the new event
                        events.append(event)

                        # Write updated events to file
                        with open(
                            event_file, "w", encoding="utf-8"
                        ) as f:
                            json.dump(events, f, indent=4)

                        print("Event saved to:", event_file)

                        # Prevent repeated alerts for this stay
                        alerted_ids.add(track_id)

                else:
                    # Person has left the restricted area
                    inside_zone_ids.discard(track_id)
                    zone_entry_times.pop(track_id, None)
                    alerted_ids.discard(track_id)

            # --------------------------------
            # 3. CLEAN UP LOST TRACKS
            # --------------------------------

            # Reset timers for people no longer detected
            for track_id in list(zone_entry_times.keys()):
                if track_id not in detected_person_ids:
                    inside_zone_ids.discard(track_id)
                    zone_entry_times.pop(track_id, None)
                    alerted_ids.discard(track_id)

            # --------------------------------
            # 4. DRAW TRACKING RESULTS
            # --------------------------------

            # Draw bounding boxes and tracking IDs
            annotated_frame = results.plot()

            # Draw virtual counting line in yellow
            cv2.line(
                annotated_frame,
                (0, current_line_y),
                (frame_width, current_line_y),
                (0, 255, 255),
                2
            )

            # Draw restricted area in red
            cv2.rectangle(
                annotated_frame,
                (zx1, zy1),
                (zx2, zy2),
                (0, 0, 255),
                2
            )

            # Label restricted area
            cv2.putText(
                annotated_frame,
                "RESTRICTED AREA",
                (zx1, max(25, zy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            # Display total people count
            cv2.putText(
                annotated_frame,
                f"People entered: {people_entered}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Print detection data every 30 frames
            if frame_count % 30 == 0:
                print("Frame:", frame_count)
                print("Detections:", detections)

            # Display camera feed
            cv2.imshow(
                "Vision Language Intelligence System",
                annotated_frame
            )

            # Press q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Release resources
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
