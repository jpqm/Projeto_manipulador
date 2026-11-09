import numpy as np

from ik_craig import de


def calculo_pontos(P0, P3, Ri, Rf, num_pontos=21, recuar_inicio=True):
    # """Gera 21 pontos de uma curva Bézier cúbica entre P0 e P3, com as orientações Ri/Rf."""
    # P0w = P0 - de*Ri[:, -1]
    # P3w = P3 - de*Rf[:, -1]

    # if (Ri == Rf).all():
    #     P1 = np.array([P0w[0], P0w[1], 100])
    #     P2 = np.array([P3w[0], P3w[1], 100])
    # elif Rf[2,2] == -1:
    #     P1 = np.array([P0w[0], P3w[1], P0w[2]])
    #     P2 = np.array([P3w[0], P3w[1], P0w[2]])
    # elif Rf[2,1] == 1:
    #     P1 = np.array([P0w[0], P0w[1], P3w[2]])
    #     P2 = np.array([P0w[0], P3w[1], P3w[2]])
    # elif Rf[2,0] == 1 and Ri[2,2] == -1:
    #     P1 = np.array([P0w[0], P0w[1], P3w[2]])
    #     P2 = np.array([P3w[0], P0w[1], P3w[2]])
    # else:
    #     P1 = np.array([(P0w[0] + P3w[0])/2, (P3w[1] + P0w[1])/2, P0w[2]])
    #     P2 = np.array([P3w[0], (P3w[1] + P0w[1])/2, (P0w[2] + P3w[2])/2])

    # t = np.linspace(0, 1, 21)[:, None]
    # pontos = (1-t)**3*P0w + 3*t*(1-t)**2*P1 + 3*t**2*(1-t)*P2 + t**3*P3w
    # return pontos[:, 0], pontos[:, 1], pontos[:, 2]

    """
    Gera pontos de uma curva Bézier cúbica generalizada usando os vetores de aproximação.
    """
    # Vetores Z do efetuador (direção de aproximação/saída)
    vetor_aprox_i = Ri[:, -1]
    vetor_aprox_f = Rf[:, -1]

    # Posições do punho (Wrist)
    P0w = P0 - de * vetor_aprox_i
    P3w = P3 - de * vetor_aprox_f

    # Distância linear entre o início e o fim
    dist = np.linalg.norm(P3w - P0w)
    
    # Fator de curvatura 'k' (determina o quão "longe" os pontos P1 e P2 vão).
    # Usamos uma proporção da distância total (ex: 0.4). 
    # O max() garante que o robô recue pelo menos uma distância mínima (ex: 50mm) 
    # mesmo se P0 e P3 estiverem muito próximos.
    k = max(dist * 0.4, 50.0) 

    # Generalização: recua P1 na direção oposta ao vetor Z inicial
    P1 = (P0w - k * vetor_aprox_i) if recuar_inicio else P0w

    # Generalização: recua P2 na direção oposta ao vetor Z final
    P2 = P3w - k * vetor_aprox_f

    # Cálculo da Bézier
    t = np.linspace(0, 1, num_pontos)[:, None]
    pontos = (1-t)**3 * P0w + 3*t*(1-t)**2 * P1 + 3*t**2*(1-t) * P2 + t**3 * P3w
    
    return pontos[:, 0], pontos[:, 1], pontos[:, 2]

def calculo_linear(P0, P3, R, num_pontos=21):
    """Gera uma interpolação linear entre dois pontos."""
    pontos = np.linspace(P0 - de*R[:, -1], P3 - de*R[:, -1], num_pontos)
    return pontos[:, 0], pontos[:, 1], pontos[:, 2]
