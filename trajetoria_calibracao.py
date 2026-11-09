import os
import json
import numpy as np
import matplotlib.pyplot as plt
from robot_control import calc_semi_esfera

base_offset = np.array([-137, 645, 25])
c = np.array([-300, 210, 0]) - base_offset
p_home = np.array([403.3643, 0, 570.3432])

# Carrega poses reais salvas ou gera pontos teóricos da rotina
json_path = os.path.join("capturas", "poses_calibracao.json")
if os.path.exists(json_path):
    with open(json_path, encoding="utf-8") as f:
        dados = json.load(f)
    pontos = np.array([np.array(pose["T_flange_base"])[:3, 3] for pose in dados.values()])
else:
    pontos = calc_semi_esfera(c)

trajetoria = np.vstack(([p_home], pontos, [p_home]))

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(0, 0, 0, color="black", s=80, label="Base (0,0,0)")
ax.scatter(c[0], c[1], c[2], color="red", s=120, marker="*", label="Alvo na Mesa")
ax.scatter(p_home[0], p_home[1], p_home[2], color="purple", s=90, marker="s", label="Home")

ax.plot(trajetoria[:, 0], trajetoria[:, 1], trajetoria[:, 2], "b--", alpha=0.7, label="Trajetória")

for i, p in enumerate(pontos):
    ax.scatter(p[0], p[1], p[2], color="blue", s=30)
    ax.text(p[0], p[1], p[2], f" {i}", fontsize=8)
    mira = (c - p) / np.linalg.norm(c - p) * 60
    ax.quiver(p[0], p[1], p[2], mira[0], mira[1], mira[2], color="green", length=1.0, arrow_length_ratio=0.3)

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_title("Trajetória da Calibração")
ax.legend()
plt.tight_layout()
plt.show()
