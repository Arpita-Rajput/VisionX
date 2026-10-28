
from ultralytics import YOLO


class YOLODetector:

    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(
            frame,
            verbose=False
        )
        return results

    def draw_detections(self, frame, results):
        annotated_frame = results[0].plot()
        return annotated_frame

    def track_objects(self, frame):
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )
        return results[0]

    def extract_tracking_data(self, results):
        detections = []

        boxes = results.boxes

        if boxes is None or boxes.id is None:
            return detections

        for box, track_id, class_id, confidence in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.id.int().cpu().tolist(),
            boxes.cls.int().cpu().tolist(),
            boxes.conf.cpu().tolist()
        ):
            detections.append({
                "track_id": track_id,
                "class_name": results.names[class_id],
                "confidence": round(confidence, 2),
                "bbox": box
            })

        return detections
