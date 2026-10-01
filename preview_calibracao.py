import numpy as np
import matplotlib.pyplot as plt
from robot_control import calc_semi_esfera

base_offset = np.array([-137, 645, 25])
c = np.array([-300, 210, 0]) - base_offset

# Gera os pontos da semi-esfera
pontos = calc_semi_esfera(c)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# 1. Base do robô e Centro do alvo na mesa
ax.scatter(0, 0, 0, color='black', s=80, label='Base do Robô (0,0,0)')
ax.scatter(c[0], c[1], c[2], color='red', s=120, marker='*', label='Centro na Mesa (Alvo)')

# 2. Trajetória conectada em ordem sequencial
ax.plot(pontos[:, 0], pontos[:, 1], pontos[:, 2], 'b--', alpha=0.7, label='Trajetória')

# 3. Posições e vetores de visão da câmera apontando para o alvo
for i, p in enumerate(pontos):
    ax.scatter(p[0], p[1], p[2], color='blue', s=40)
    ax.text(p[0], p[1], p[2], f" {i}", fontsize=9)
    mira = c - p
    mira = mira / np.linalg.norm(mira) * 60
    ax.quiver(p[0], p[1], p[2], mira[0], mira[1], mira[2], color='green', length=1.0, arrow_length_ratio=0.3)

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Pré-visualização 3D da Rotina de Calibração')
ax.legend()
plt.tight_layout()
plt.show()
