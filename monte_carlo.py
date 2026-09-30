import numpy as np
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import ik_craig as ik

# 1. Definir os elos usando RevoluteMDH (Parâmetros de Craig)
# A assinatura recebe: a (que representa a_{i-1}), alpha (alpha_{i-1}), d (d_i) e o limite da junta
# Substitua os valores pelos do seu gêmeo digital/projeto
link1 = rtb.RevoluteMDH(a=ik.a1_1,    alpha=np.deg2rad(ik.alpha1_1),  d=ik.d1,    qlim=[0, 2*np.pi])
link2 = rtb.RevoluteMDH(a=ik.a2_1,    alpha=np.deg2rad(ik.alpha2_1),  d=ik.d2,    qlim=[np.deg2rad(-75), np.deg2rad(125)], offset=-np.pi/2)
link3 = rtb.RevoluteMDH(a=ik.a3_1,    alpha=np.deg2rad(ik.alpha3_1),  d=ik.d3,    qlim=[np.deg2rad(-50), np.deg2rad(100)])
link4 = rtb.RevoluteMDH(a=ik.a4_1,    alpha=np.deg2rad(-90),          d=ik.d4,    qlim=[0, 2*np.pi])
link5 = rtb.RevoluteMDH(a=0,          alpha=np.deg2rad(90),           d=0,        qlim=[-np.pi/4, np.pi/4])
link6 = rtb.RevoluteMDH(a=0,          alpha=np.deg2rad(-90),          d=ik.de,    qlim=[0, 2*np.pi])

# Criar o robô 6-DOF
robot_mdh = rtb.DHRobot([link1, link2, link3, link4, link5, link6], name="Manipulador_MDH")

# 2. Configurar o Método de Monte Carlo
num_pontos = 200000
q_aleatorios = np.zeros((num_pontos, 6))

# Gerar ângulos aleatórios vetorizados respeitando os limites mecânicos
for i in range(6):
    q_min, q_max = robot_mdh.links[i].qlim
    q_aleatorios[:, i] = np.random.uniform(q_min, q_max, num_pontos)

# 3. Cinemática Direta vetorizada para toda a matriz (O gargalo computacional é resolvido aqui)
# fkine retorna um objeto SE3 do pacote spatialmath
poses = robot_mdh.fkine(q_aleatorios)

# 4. Extrair as coordenadas de translação X, Y, Z do efetuador
x = poses.t[:, 0]
y = poses.t[:, 1]
z = poses.t[:, 2]

# 5. Gerar e salvar Voxel Grid
voxel_size = 5.0  # Resolução do cubo em mm
pontos = np.column_stack((x, y, z))
min_bound = np.floor(pontos.min(axis=0))
max_bound = np.ceil(pontos.max(axis=0))
dimensoes = ((max_bound - min_bound) / voxel_size).astype(int) + 1

grid = np.zeros(dimensoes, dtype=bool)
indices = ((pontos - min_bound) / voxel_size).astype(int)
grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

np.savez_compressed("workspace_voxel.npz", grid=grid, min_bound=min_bound, voxel_size=voxel_size)
print(f"Voxel Grid salvo ({grid.shape}, resolução {voxel_size} mm).")

# 5. Visualização com Matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotagem da nuvem de pontos com mapeamento de cor no eixo Z para melhor percepção de profundidade
scatter = ax.scatter(x, y, z, s=0.5, c=z, cmap='plasma', alpha=0.3)
fig.colorbar(scatter, label='Altura Z (mm)')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Workspace com DH Modificado (Monte Carlo)')

# Ajustar proporção dos eixos para não distorcer o formato do workspace
ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])
plt.show()