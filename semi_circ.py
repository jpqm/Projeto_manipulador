import numpy as np

def calc_semi_circ(c, u, v):
    r = 500
    
    t = np.linspace(0, 1, 20)

    # Convertendo para array caso venham como listas
    u = np.array(u, dtype=float)
    v = np.array(v, dtype=float)
    c = np.array(c, dtype=float)

    # 1. Normalização simplificada 
    u = u / np.linalg.norm(u)
    
    # 2. Ortogonalização (Gram-Schmidt) garante que v é estritamente perpendicular a u
    v = v - np.dot(v, u) * u
    v = v / np.linalg.norm(v)

    # 3. Solução do erro de Broadcasting: (20,) -> (20, 1)
    t_col = t[:, np.newaxis]

    # 4. Uso de np.pi (meia volta) em vez de 2*np.pi (volta inteira)
    pontos = c + r * np.cos(np.pi * t_col) * u + r * np.sin(np.pi * t_col) * v

    return pontos[:, 0], pontos[:, 1], pontos[:, 2]