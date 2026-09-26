from pathlib import Path
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).parent / "yolo" / "pesos" / "last.pt"


class YOLODetector:
    def __init__(self, weights_path=DEFAULT_WEIGHTS, conf=0.5):
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Pesos do YOLO nao encontrados em: {weights_path}")
        self.model = YOLO(weights_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        result = results[0]
        annotated_frame = result.plot()

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "class": cls_name,
                "confidence": confidence,
                "bbox": xyxy
            })
        return annotated_frame, detections
