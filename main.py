
import cv2
import os
import time

from camera.camera_manager import CameraManager
from detection.yolo_detector import YOLODetector
from event_detection import RestrictedAreaEventDetector


def main():

    # ==========================================
    # 1. INITIALIZE CAMERA AND YOLO
    # ==========================================

    camera = CameraManager(0)

    detector = YOLODetector()

    frame_count = 0

    # ==========================================
    # 2. PEOPLE COUNTING SETTINGS
    # ==========================================

    line_y = 300

    people_entered = 0

    previous_positions = {}

    counted_ids = set()

    # ==========================================
    # 3. RESTRICTED AREA COORDINATES
    # ==========================================

    zone_x1 = 200
    zone_y1 = 100

    zone_x2 = 500
    zone_y2 = 400

    # ==========================================
    # 4. INITIALIZE EVENT DETECTOR
    # ==========================================

    event_detector = RestrictedAreaEventDetector(

        zone_x1=zone_x1,
        zone_y1=zone_y1,

        zone_x2=zone_x2,
        zone_y2=zone_y2,

        camera_id=1,

        # Loitering threshold: 120 seconds
        loitering_threshold=120
    )

    # ==========================================
    # 5. NATURAL LANGUAGE MONITORING
    # ==========================================

    print("\n==========================================")
    print("       VLIS MONITORING SYSTEM")
    print("==========================================")

    print("\nAvailable monitoring examples:")

    print("1. Alert me if anyone enters the restricted area.")
    print("2. Alert me if anyone exits the restricted area.")
    print("3. Alert me if anyone stays longer than 2 minutes.")
    print("4. Monitor entry, exit and loitering activities.")

    user_prompt = input(
        "\nEnter your monitoring instruction: "
    )

    # Configure monitoring rules
    event_detector.configure_from_prompt(user_prompt)

    # ==========================================
    # 6. START CAMERA
    # ==========================================

    print("\nCamera started...")
    print("Press 'q' to quit.")

    try:

        while True:

            # ----------------------------------
            # READ CAMERA FRAME
            # ----------------------------------

            frame = camera.read_frame()

            if frame is None:

                print("Unable to read frame from camera.")

                break

            frame_count += 1

            frame_height, frame_width = frame.shape[:2]

            # ----------------------------------
            # ADJUST COUNTING LINE
            # ----------------------------------

            current_line_y = min(
                max(line_y, 0),
                frame_height - 1
            )

            # ----------------------------------
            # ADJUST RESTRICTED AREA
            # ----------------------------------

            zx1 = max(
                0,
                min(zone_x1, frame_width - 1)
            )

            zy1 = max(
                0,
                min(zone_y1, frame_height - 1)
            )

            zx2 = max(
                0,
                min(zone_x2, frame_width - 1)
            )

            zy2 = max(
                0,
                min(zone_y2, frame_height - 1)
            )

            # ==================================
            # 7. OBJECT DETECTION AND TRACKING
            # ==================================

            results = detector.track_objects(frame)

            detections = detector.extract_tracking_data(
                results
            )

            # ==================================
            # 8. VLIS EVENT DETECTION ENGINE
            # ==================================

            # Process:
            # - Restricted area entry
            # - Restricted area exit
            # - Loitering detection
            # - Event history logging

            event_detector.process_detections(
                detections
            )

            # ==================================
            # 9. PEOPLE COUNTING
            # ==================================

            detected_person_ids = set()

            for detection in detections:

                if detection["class_name"] != "person":
                    continue

                track_id = detection["track_id"]

                if track_id is None:
                    continue

                track_id = int(track_id)

                x1, y1, x2, y2 = detection["bbox"]

                detected_person_ids.add(track_id)

                # Bottom-center position for line crossing
                center_x = int((x1 + x2) / 2)

                bottom_y = int(y2)

                # Check previous position
                if track_id in previous_positions:

                    previous_y = previous_positions[track_id]

                    # Count top-to-bottom crossing once
                    if (
                        previous_y < current_line_y
                        and bottom_y >= current_line_y
                        and track_id not in counted_ids
                    ):

                        people_entered += 1

                        counted_ids.add(track_id)

                        print(
                            f"Person ID {track_id} crossed "
                            f"the counting line. "
                            f"Total entered: {people_entered}"
                        )

                previous_positions[track_id] = bottom_y

            # ==================================
            # 10. CLEAN UP LOST TRACK POSITIONS
            # ==================================

            for track_id in list(previous_positions.keys()):

                if track_id not in detected_person_ids:

                    previous_positions.pop(
                        track_id,
                        None
                    )

            # ==================================
            # 11. DRAW TRACKING RESULTS
            # ==================================

            annotated_frame = results.plot()

            # ----------------------------------
            # DRAW COUNTING LINE
            # ----------------------------------

            cv2.line(

                annotated_frame,

                (0, current_line_y),

                (frame_width, current_line_y),

                (0, 255, 255),

                2
            )

            # ----------------------------------
            # DRAW RESTRICTED AREA
            # ----------------------------------

            cv2.rectangle(

                annotated_frame,

                (zx1, zy1),

                (zx2, zy2),

                (0, 0, 255),

                2
            )

            cv2.putText(

                annotated_frame,

                "RESTRICTED AREA",

                (zx1, max(25, zy1 - 10)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 0, 255),

                2
            )

            # ==================================
            # 12. DRAW PERSON CENTER POINTS
            # ==================================

            for detection in detections:

                if detection["class_name"] != "person":
                    continue

                x1, y1, x2, y2 = detection["bbox"]

                center_x = int((x1 + x2) / 2)

                center_y = int((y1 + y2) / 2)

                # Purple center point
                cv2.circle(

                    annotated_frame,

                    (center_x, center_y),

                    6,

                    (255, 0, 255),

                    -1
                )

            # ==================================
            # 13. DISPLAY PEOPLE COUNT
            # ==================================

            cv2.putText(

                annotated_frame,

                f"People entered: {people_entered}",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2
            )

            # ==================================
            # 14. PRINT DETECTION INFORMATION
            # ==================================

            if frame_count % 30 == 0:

                print("\nFrame:", frame_count)

                print("Detections:", detections)

            # ==================================
            # 15. DISPLAY CAMERA FEED
            # ==================================

            cv2.imshow(

                "Vision Language Intelligence System",

                annotated_frame
            )

            # ==================================
            # 16. EXIT CAMERA
            # ==================================

            if cv2.waitKey(1) & 0xFF == ord("q"):

                break

    except KeyboardInterrupt:

        print("\nCamera stopped by user.")

    finally:

        # ======================================
        # 17. DISPLAY EVENT HISTORY SUMMARY
        # ======================================

        print("\n")
        print("=" * 50)

        print("       VLIS EVENT HISTORY SUMMARY")

        print("=" * 50)

        try:

            summary = event_detector.get_event_summary()

            print(
                "Total Events:",
                summary["total_events"]
            )

            print(
                "Total Entries:",
                summary["entries"]
            )

            print(
                "Total Exits:",
                summary["exits"]
            )

            print(
                "Loitering Alerts:",
                summary["loitering_alerts"]
            )

            print("\nEvent history saved at:")

            print("output/events.json")

        except Exception as e:

            print("Unable to generate event summary:", e)

        print("=" * 50)

        # ======================================
        # 18. RELEASE CAMERA RESOURCES
        # ======================================

        camera.release()

        cv2.destroyAllWindows()

        print("\nVLIS monitoring stopped.")


if __name__ == "__main__":

    main()
