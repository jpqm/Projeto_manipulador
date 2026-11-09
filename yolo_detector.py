from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

DEFAULT_WEIGHTS = Path(__file__).parent / "yolo" / "pesos" / "last.pt"
CALIB_INTRINSECA = Path(__file__).parent / "calibracao" / "calibracao_intrinseca.npz"
ALTURA_MM = {"Tampa-porca": 10.0, "base": 40.0, "sensor": 25.0}


class YOLODetector:
    def __init__(self, weights_path=DEFAULT_WEIGHTS, conf=0.85):
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Pesos do YOLO nao encontrados em: {weights_path}")
        self.model = YOLO(weights_path)
        self.conf = conf
        if CALIB_INTRINSECA.exists():
            calib = np.load(CALIB_INTRINSECA)
            self.K = calib["K"].astype(np.float32)
            self.dist = calib["dist"].astype(np.float32)
        else:
            self.K = None
            self.dist = None
        p = cv2.aruco.DetectorParameters()
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 23
        p.adaptiveThreshWinSizeStep = 4                # inclui janelas 7 e 11 px (essenciais para marcador de ~22 px)
        p.adaptiveThreshConstant = 5                   # 3 é baixo para canto sombreado; ajuste entre 3 e 7
        p.perspectiveRemovePixelPerCell = 8            # padrão 4; dobra a resolução ao ler células pequenas
        p.perspectiveRemoveIgnoredMarginPerCell = 0.2  # 0.3 era alto demais para célula de 3,5 px
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), p)

    def detect(self, frame):
        allowed_classes = [k for k, v in self.model.names.items() if v != "base"]
        results = self.model.predict(frame, conf=self.conf, classes=allowed_classes, verbose=False)
        result = results[0]
        annotated_frame = result.plot()

        detections = []
        has_masks = result.masks is not None and len(result.masks) == len(result.boxes)
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            det_data = {
                "class": cls_name,
                "confidence": confidence,
                "bbox": xyxy
            }
            if has_masks and len(result.masks.xy[i]) >= 3:
                det_data["mask_pts"] = result.masks.xy[i]
                vx, vy, x0, y0 = cv2.fitLine(result.masks.xy[i], cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                cv2.line(annotated_frame, (int(x0 - vx * 60), int(y0 - vy * 60)), (int(x0 + vx * 60), int(y0 + vy * 60)), (0, 255, 0), 2)
            detections.append(det_data)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)
            cfg = {
                3: (2, (-9.14, 8, 0.0)),
                1: (0, (-514.56, 141.4, 0.0)),
                2: (0, (-7.72, 456.5, 0.0)),
                0: (2, (-501, 500.5, 0.0))
            }
            img_pts, obj_pts = [], []
            for c, i in zip(corners, ids.flatten()):
                if i in cfg:
                    idx, phys_coord = cfg[i]
                    pt = c[0][idx]
                    img_pts.append(pt)
                    obj_pts.append(phys_coord)
                    cv2.putText(annotated_frame, f"({phys_coord[0]:.1f}, {phys_coord[1]:.1f})", (int(pt[0]), int(pt[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            
            if len(img_pts) == 4:
                img_pts = np.array(img_pts, dtype=np.float32)
                obj_pts = np.array(obj_pts, dtype=np.float32)
                und = lambda p: cv2.undistortPoints(np.float32(p).reshape(-1, 1, 2), self.K, self.dist, P=self.K).reshape(-1, 2)
                H = cv2.getPerspectiveTransform(und(img_pts), obj_pts[:, :2].copy())
                _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
                C = (-cv2.Rodrigues(rvec)[0].T @ tvec).ravel()  # centro óptico no frame da mesa

                for det in detections:
                    m = cv2.moments(det["mask_pts"].astype(np.float32)) if "mask_pts" in det else {"m00": 0}
                    if m["m00"] > 0:
                        u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
                    else:
                        u, v = (det["bbox"][0] + det["bbox"][2]) / 2.0, (det["bbox"][1] + det["bbox"][3]) / 2.0
                    p0 = np.array([*cv2.perspectiveTransform(und([u, v])[None], H)[0, 0], 0.0])
                    zc = ALTURA_MM.get(det["class"], 10.0) * 0.5  # ponytail: centroide visual ~ meia-altura; ajuste o 0.5 (0.5 a 1.0 conforme o quanto o topo domina a máscara)
                    p = p0 + (C - p0) * zc / abs(C[2])          # desliza pelo raio da câmera até a altura zc
                    det["phys_center"] = [float(p[0]), float(p[1])]
                    if "mask_pts" in det:
                        pts_real = cv2.perspectiveTransform(und(det["mask_pts"])[None], H)[0]
                        vx, vy, _, _ = cv2.fitLine(pts_real, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                        det["angle"] = float(np.degrees(np.arctan2(vy, vx)))

                for c, i in zip(corners, ids.flatten()):
                    if 4 <= i <= 8:
                        X, Y = cv2.perspectiveTransform(und(c[0])[None], H)[0].mean(axis=0)
                        detections.append({"class": f"aruco_{i}", "confidence": 1.0, "phys_center": [float(X), float(Y)]})
                        u, v = c[0].mean(axis=0)
                        cv2.putText(annotated_frame, f"{i}:({X:.1f},{Y:.1f})", (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            else:
                faltando = sorted(set(cfg.keys()) - set(ids.flatten()))
                print(f"[Aruco] Falha no cálculo: {len(img_pts)}/4 marcadores detectados. Faltam IDs: {faltando}")
        else:
            print("[Aruco] Falha no cálculo: nenhum marcador detectado na cena.")

        return annotated_frame, detections
