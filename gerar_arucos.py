import cv2
import numpy as np
from PIL import Image

canvas = np.ones((3508, 2480), dtype=np.uint8) * 255
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

px, py = (2480 - 1600) // 3, (3508 - 1600) // 3
for i in range(4):
    y, x = py + (i // 2) * (800 + py), px + (i % 2) * (800 + px)
    canvas[y:y+800, x:x+800] = cv2.aruco.generateImageMarker(d, i, 800)

Image.fromarray(canvas).save("arucos_a4.pdf", resolution=300.0)

board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.03, d)
charuco_img = board.generateImage((2480, 3508), marginSize=100)
Image.fromarray(charuco_img).save("charuco_a4.pdf", resolution=300.0)

