from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

DEFAULT_WEIGHTS = Path(__file__).parent / "yolo" / "pesos" / "last.pt"


class YOLODetector:
    def __init__(self, weights_path=DEFAULT_WEIGHTS, conf=0.5):
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Pesos do YOLO nao encontrados em: {weights_path}")
        self.model = YOLO(weights_path)
        self.conf = conf

    def detect(self, frame):
        allowed_classes = [k for k, v in self.model.names.items()]
        results = self.model.predict(frame, conf=self.conf, classes=allowed_classes, verbose=False)
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
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fundo = cv2.GaussianBlur(gray_frame, (81, 81), 0)
        gray_clahe = cv2.divide(gray_frame, fundo, scale=255)

        # 3. Aplicar Filtro de Mediana para remover os buracos brancos (reflexos)
        # O kernel (ex: 5) deve ser sempre um número ímpar. 
        # Se o ponto branco for grande, aumente para 7 ou 9.
        gray_tratado = cv2.medianBlur(gray_clahe, 5)

        # (Opcional) 4. Operação Morfológica se o Filtro de Mediana não for suficiente
        kernel = np.ones((3,3), np.uint8)
        gray_tratado = cv2.morphologyEx(gray_tratado, cv2.MORPH_OPEN, kernel)

        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        # Simulando a binarizacao que o ArUco faz internamente para visualizacao
        bin_img = cv2.adaptiveThreshold(gray_tratado, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 7)
        cv2.imwrite("binarizada.jpg", bin_img)
        print(f"Img binarizada: shape={bin_img.shape} (Alt x Larg), dtype={bin_img.dtype} (Inteiro 8 bits, 0=preto/255=branco)")

        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 35
        params.adaptiveThreshWinSizeStep = 3
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        corners, ids, _ = cv2.aruco.ArucoDetector(d, params).detectMarkers(gray_clahe)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)
            cfg = {
                1: (3, (-12.5, 8.3, 0.0)),
                0: (2, (-621.15, 142.5, 0.0)),
                3: (0, (11.15, 445.0, 0.0)),
                2: (1, (-550.0, 487.35, 0.0))
            }
            img_pts, obj_pts = [], []
            for c, i in zip(corners, ids.flatten()):
                if i in cfg:
                    idx, phys_coord = cfg[i]
                    pt = c[0][idx]
                    img_pts.append(pt)
                    obj_pts.append(phys_coord)
            
            if len(img_pts) == 4:
                img_pts = np.array(img_pts, dtype=np.float32)
                obj_pts = np.array(obj_pts, dtype=np.float32)

                # Homografia 2D
                H, _ = cv2.findHomography(img_pts, obj_pts[:, :2])

                for det in detections:
                    cx, cy = (det["bbox"][0] + det["bbox"][2]) / 2, (det["bbox"][1] + det["bbox"][3]) / 2
                    phys_center = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), H)[0][0]
                    det["phys_center"] = phys_center.tolist()

                # Pose 3D (solvePnP é mais direto que decompor homografia)
                h_img, w_img = frame.shape[:2]
                K = np.array([[w_img, 0, w_img/2], [0, w_img, h_img/2], [0, 0, 1]], dtype=np.float32) # Matriz intrínseca assumida
                _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, None)

                R, _ = cv2.Rodrigues(rvec)
                pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                yaw = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
                roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            else:
                faltando = sorted(set(cfg.keys()) - set(ids.flatten()))
                print(f"[Aruco] Falha no cálculo: {len(img_pts)}/4 marcadores detectados. Faltam IDs: {faltando}")
        else:
            print("[Aruco] Falha no cálculo: nenhum marcador detectado na cena.")

        return annotated_frame, detections
