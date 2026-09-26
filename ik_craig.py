import numpy as np

a1_1 = 0
a2_1 = 20.3563
a3_1 = 261.01
a4_1 = 21.7052
a5_1 = 0
a6_1 = 0
d1 = 287.628
d2 = 0
d3 = 0
d4 = 264.193
d5 = 0
d6 = 0
de = 118.815
alpha1_1 = 0
alpha2_1 = -90
alpha3_1 = 0
alpha4_1 = -90
alpha5_1 = 90
alpha6_1 = -90

# Matriz homogênea para DH (Craig)
def matriz_trans_np(d, a, theta, alpha):
    """Monta a matriz de transformação homogênea 4x4 a partir dos parâmetros DH (Craig)."""
    alpha = np.deg2rad(alpha)
    return np.matrix([[np.cos(theta), -np.sin(theta), 0, a],
                      [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                      [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), np.cos(alpha)*d],
                      [0, 0, 0, 1]])

np.set_printoptions(suppress=True, precision=3)

def limpar_matriz(M, tol=1e-10):
    """Zera valores muito pequenos da matriz para evitar ruído numérico."""
    M[np.abs(M) < tol] = 0
    return M

def calculo_angulos(x, y, z):
    """Cinemática inversa de posição: retorna os ângulos das juntas 1, 2 e 3 (em graus)."""
    theta1 = np.atan2(y, x)

    T1 = matriz_trans_np(d1, a1_1, theta1, alpha1_1)
    T1 = limpar_matriz(T1)

    T1_inv = np.linalg.inv(T1)
    T1_inv = limpar_matriz(T1_inv)

    x1 = T1_inv[0,0]*x + T1_inv[0,1]*y + T1_inv[0,2]*z + T1_inv[0,3]
    y1 = T1_inv[1,0]*x + T1_inv[1,1]*y + T1_inv[1,2]*z + T1_inv[1,3]
    z1 = T1_inv[2,0]*x + T1_inv[2,1]*y + T1_inv[2,2]*z + T1_inv[2,3]

    K = (np.pow(x1, 2) + np.pow(z1, 2) + np.pow(a2_1, 2) - 2*x1*a2_1 - np.pow(a3_1, 2) - np.pow(a4_1, 2) - np.pow(d4, 2))/(2*a3_1)

    theta3 = np.atan2(a4_1, d4) - np.atan2(K, np.sqrt(np.pow(a4_1, 2) + np.pow(d4, 2) - np.pow(K, 2)))

    M = np.cos(theta1)*x + np.sin(theta1)*y - a2_1
    N = d1 - z

    s23 = (a4_1 + a3_1*np.cos(theta3))*N + (-d4 + a3_1*np.sin(theta3))*M
    c23 = (a4_1 + a3_1*np.cos(theta3))*M - (-d4 + a3_1*np.sin(theta3))*N

    theta23 = np.atan2(s23, c23)

    theta2 = theta23 - theta3

    theta1 = np.round(np.rad2deg(theta1), 2)
    theta2 = np.round(np.rad2deg(theta2) + 90, 2)
    theta3 = np.round(np.rad2deg(theta3), 2)

    return theta1, theta2, theta3

def calculo_angulos_abc(R, P, compensar_de=True):
    """Cinemática inversa de orientação: retorna os ângulos A, B e C (juntas 4, 5 e 6)."""
    if compensar_de:
        P = P - de*R[:,-1].T

    P = P.reshape(3,1)

    topo = np.hstack((R, P))
    base = np.array([[0,0,0,1]])

    T06 = np.vstack((topo, base))

    Px = T06[0,3]
    Py = T06[1,3]
    Pz = T06[2,3]

    theta1 = np.arctan2(Py, Px)

    T1 = matriz_trans_np(d1, a1_1, theta1, alpha1_1)

    T16 = np.linalg.inv(T1) @ T06

    Px16 = T16[0,3]
    Pz16 = T16[2,3]

    K = (np.power(Px16, 2) + np.power(Pz16, 2) + np.power(a2_1, 2) - 2*Px16*a2_1 - np.power(a3_1, 2) - np.power(a4_1, 2) - np.power(d4, 2))/(2*a3_1)
    
    theta3 = np.arctan2(a4_1, d4) - np.arctan2(K, np.sqrt(np.power(a4_1, 2) + np.power(d4, 2) - np.power(K, 2)))

    M = np.cos(theta1)*Px + np.sin(theta1)*Py - a2_1
    N = d1 - Pz

    s23 = (a4_1 + a3_1*np.cos(theta3))*N + (-d4 + a3_1*np.sin(theta3))*M
    c23 = (a4_1 + a3_1*np.cos(theta3))*M - (-d4 + a3_1*np.sin(theta3))*N

    theta23 = np.arctan2(s23, c23)

    theta2 = theta23 - theta3

    T2 = matriz_trans_np(d2, a2_1, theta2, alpha2_1)
    T3 = matriz_trans_np(d3, a3_1, theta3, alpha3_1)

    T03 = T1 @ T2 @ T3
    T36 = np.linalg.inv(T03) @ T06
    T36 = limpar_matriz(T36)

    c5 = T36[1,2]

    if np.isclose(abs(c5), 1):
        theta4 = 0.0
        theta5 = 0.0
    else:
        u = np.cos(theta1)*np.cos(theta23)*T06[0,2] + np.sin(theta1)*np.cos(theta23)*T06[1,2] - np.sin(theta23)*T06[2,2]
        v = -np.sin(theta1)*T06[0,2] + np.cos(theta1)*T06[1,2]
        theta4 = np.arctan2(v, -u)
        s5 = T06[2,2]*np.sin(theta23)*np.cos(theta4) - T06[0,2]*(np.cos(theta1)*np.cos(theta23)*np.cos(theta4) + np.sin(theta1)*np.sin(theta4)) - T06[1,2]*(np.sin(theta1)*np.cos(theta23)*np.cos(theta4) - np.cos(theta1)*np.sin(theta4))
        c5 = -T06[0,2]*np.cos(theta1)*np.sin(theta23) - T06[1,2]*np.sin(theta1)*np.sin(theta23) - T06[2,2]*np.cos(theta23)
        theta5 = np.arctan2(s5, c5)

    s6 = T06[2,0]*np.sin(theta23)*np.sin(theta4) - T06[0,0]*(np.cos(theta1)*np.cos(theta23)*np.sin(theta4) - np.sin(theta1)*np.cos(theta4)) - T06[1,0]*(np.sin(theta1)*np.cos(theta23)*np.sin(theta4) + np.cos(theta1)*np.cos(theta4))
    c6 = T06[0,0]*((np.cos(theta1)*np.cos(theta23)*np.cos(theta4) + np.sin(theta1)*np.sin(theta4))*np.cos(theta5) - np.cos(theta1)*np.sin(theta23)*np.sin(theta5)) + T06[1,0]*((np.sin(theta1)*np.cos(theta23)*np.cos(theta4) - np.cos(theta1)*np.sin(theta4))*np.cos(theta5) - np.sin(theta1)*np.sin(theta23)*np.sin(theta5)) - T06[2,0]*(np.sin(theta23)*np.cos(theta4)*np.cos(theta5) + np.cos(theta23)*np.sin(theta5))

    theta6 = np.arctan2(s6, c6)

    theta4 = np.round(np.rad2deg(theta4), 2)
    theta5 = np.round(np.rad2deg(theta5), 2)
    theta6 = np.round(np.rad2deg(theta6), 2)

    if abs(theta4) == 180 and abs(theta6) == 180:
        theta4 = 0
        theta6 = 0
        theta5 = -theta5
    elif abs(theta4) == 180:
        theta4 = 0
        theta5 = theta5 - 90
    elif abs(theta6) == 180:
        theta6 = 0

    theta5 = -theta5

    return theta4, theta5, theta6

def calculo_angulos_abc_semi_circ(R, P):
    """Cinemática inversa de orientação sem compensar o deslocamento de ponta."""
    return calculo_angulos_abc(R, P, compensar_de=False)