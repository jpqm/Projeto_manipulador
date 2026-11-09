import cv2
import numpy as np
from PIL import Image

canvas = np.ones((3508, 2480), dtype=np.uint8) * 255
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

y = 150
for mm in [20, 25, 30]:
    s = round(mm * 300 / 25.4)
    for i in range(5):  # IDs 4..8; vão de 120 px (~10 mm) para 5 caberem no A4 com o de 30 mm
        canvas[y:y+s, 100+i*(s+120):100+i*(s+120)+s] = cv2.aruco.generateImageMarker(d, 4 + i, s)
    y += s + 300

Image.fromarray(canvas).save("arucos_a4.pdf", resolution=300.0)

board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.03, d)
charuco_img = board.generateImage((2480, 3508), marginSize=100)
Image.fromarray(charuco_img).save("charuco_a4.pdf", resolution=300.0)

