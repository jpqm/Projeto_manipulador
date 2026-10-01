import os
import re
import time

import cv2

import numpy as np
import bezier as bz
import ik_craig as ik
from config import GCODE_LOG

def calc_semi_circ(c, u, v, r=500):
    t = np.linspace(0, 1, 20)[:, np.newaxis]
    u = np.asarray(u, dtype=float) / np.linalg.norm(u)
    v = np.asarray(v, dtype=float)
    v = v - np.dot(v, u) * u
    v = v / np.linalg.norm(v)
    pontos = c + r * np.cos(np.pi * t) * u + r * np.sin(np.pi * t) * v
    return pontos[:, 0], pontos[:, 1], pontos[:, 2]

def calc_semi_esfera(c, r=500, elev_range=(65, 85), azim_range=(0, 110), n_elev=5, n_azim=6):
    """Gera pontos em trajetória serpenteada sobre a semi-esfera sem tocar a mesa."""
    elevs = np.linspace(elev_range[0], elev_range[1], n_elev)
    pontos = []
    for idx, el_deg in enumerate(elevs):
        azims = np.linspace(azim_range[0], azim_range[1], n_azim)
        if idx % 2 == 1:
            azims = azims[::-1]
        el = np.deg2rad(el_deg)
        for az_deg in azims:
            az = np.deg2rad(az_deg)
            pontos.append(c + r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]))
    return np.array(pontos)

class RobotController:
    def __init__(self, serial_driver, unity_client):
        """Guarda as referências de comunicação e o estado inicial de posição e orientação do robô."""
        self.serial = serial_driver
        self.unity = unity_client
        
        # Estado inicial do manipulador
        self.P0 = np.array([403.3643, 0, 570.3432])
        self.Ri = np.array([[0, 0, 1], 
                            [0, -1, 0], 
                            [1, 0, 0]])
        self.base_offset = np.array([-137, 645, 25])
        self.modo_juntas = False
        self.R_baixo = np.array([[ 0,  -1,  0], 
                                 [ -1, 0,  0], 
                                 [ 0,  0, -1]]) 

    def interpolar_abc(self, R_ini, P_ini, R_fim, P_fim, n=21, A=None, B=None, C=None):
        """Interpola linearmente os ângulos do pulso (A, B, C) entre a pose inicial e a final."""
        if A == None and B == None and C == None:
            A0, B0, C0 = ik.calculo_angulos_abc(R_ini, P_ini)
        else:
            A0, B0, C0 = A, B, C
        Af, Bf, Cf = ik.calculo_angulos_abc(R_fim, P_fim)
        return (np.round(np.linspace(A0, Af, n), 2),
                np.round(np.linspace(B0, Bf, n), 2),
                np.round(np.linspace(C0, Cf, n), 2))

    def executar_movimento(self, x, y, z, A, B, C, feedrate=800):
        """Calcula os ângulos das juntas para cada ponto, envia ao GRBL (G1) e espelha no Unity."""
        angulos = []
        for i in range(len(x)):
            theta1, theta2, theta3 = ik.calculo_angulos(x[i], y[i], z[i])
            angulos.append([theta1, theta2, theta3, A[i], -C[i], B[i]])

        for i in range(len(angulos)):
            theta1, theta2, theta3, A_grbl, B_grbl, C_grbl = angulos[i]
            self.unity.send_angles(theta1, theta2, -theta3, -A[i], B[i], -C[i], feedrate)
            self.serial.send(f"G1 X{theta1} Y{theta2} Z{theta3} A{A_grbl} B{B_grbl} C{C_grbl} F{feedrate}")

    def ponto_alcancavel(self, ponto, R=None):
        """Retorna True se todos os 6 ângulos da cinemática inversa forem calculáveis."""
        if R is None:
            R = self.R_baixo
        try:
            with np.errstate(invalid="ignore"):
                # ponytail: valida apenas solução matemática (sem NaN); upgrade: checar qlim se colidir
                Pw = np.asarray(ponto, dtype=float) - ik.de * R[:, -1]
                t1, t2, t3 = ik.calculo_angulos(*Pw)
                t4, t5, t6 = ik.calculo_angulos_abc(R, ponto)
            return not np.any(np.isnan([t1, t2, t3, t4, t5, t6]))
        except Exception:
            return False

    def enviar_juntas(self, j1, j2, j3, j4, j5, j6):
        """Envia um G1 direto com os 6 ângulos das juntas (valores GRBL) e espelha no Unity.
        Ativa o modo juntas: bloqueia trajetórias/rotina até o Home ser usado."""
        self.modo_juntas = True
        self.serial.send(f"G1 X{j1} Y{j2} Z{j3} A{j4} B{j6} C{j5} F800")
        self.unity.send_angles(j1, j2, -j3, -j4, j5, j6, 800)

    def calcular_tempo_trajetoria(self, x, y, z, theta4, theta5, theta6, feedrate=800, fator_seg=1.1):
        """Estima o tempo (s) da trajetória pela distância percorrida em cada segmento dividida pelo feedrate."""
        angulos = []
        for i in range(21):
            t1, t2, t3 = ik.calculo_angulos(x[i], y[i], z[i])
            angulos.append([t1, t2, t3, theta4[i], theta5[i], theta6[i]])

        angulos = np.array(angulos)
        deltas = np.diff(angulos, axis=0)                              # (20, 6)
        distancia_total = np.sum(np.sqrt(np.sum(deltas**2, axis=1)))   # norma euclidiana por segmento, somada

        tempo_min = distancia_total / feedrate     # F do GRBL é sempre unidades/min
        tempo_s = tempo_min * 60 * fator_seg

        print(f"[TRAJETÓRIA] Tempo estimado: {tempo_s:.1f} s")
        return tempo_s

    def _mover_e_aguardar(self, x, y, z, A, B, C, feedrate=800):
        """Executa o movimento e aguarda o tempo estimado de percurso."""
        self.executar_movimento(x, y, z, A, B, C, feedrate=feedrate)
        pausa = self.calcular_tempo_trajetoria(x, y, z, A, B, C, feedrate=feedrate)
        time.sleep(pausa + 1)

    def home(self):
        """Retorna o robô à posição inicial (Home) com trajetória Bézier e atualiza o estado."""
        if self.modo_juntas:
            print("Modo juntas ativo — desfazendo último movimento via log (Home).")
            self.recuperar_do_log()
            self.P0 = np.array([403.3643, 0, 570.3432])
            self.Ri = np.array([[0, 0, 1], 
                                [0, -1, 0], 
                                [1, 0, 0]])
            self.modo_juntas = False
            return None, None, None

        P3 = np.array([403.3643, 0, 570.3432])
        Rf = np.array([[0, 0, 1], 
                       [0, -1, 0], 
                       [1, 0, 0]])
        
        x1, y1, z1 = bz.calculo_pontos(self.P0, P3, self.Ri, Rf)
        A1, B1, C1 = self.interpolar_abc(self.Ri, self.P0, Rf, P3, 21)
        
        self.executar_movimento(x1, y1, z1, A1, B1, C1)
        self.Ri = Rf
        self.P0 = P3

        return x1, y1, z1

    def recuperar_do_log(self):
        """Se o último G1 do log não for tudo zero, envia o inverso para
        desfazer o deslocamento e reancora o zero (G92)."""
        try:
            with open(GCODE_LOG) as f:
                linhas = f.read().splitlines()
        except (FileNotFoundError, OSError):
            return

        padrao = re.compile(r'G1\s+X(-?[\d.]+)\s+Y(-?[\d.]+)\s+Z(-?[\d.]+)\s+A(-?[\d.]+)\s+B(-?[\d.]+)\s+C(-?[\d.]+)')
        ultimo = None
        for linha in reversed(linhas):
            m = padrao.search(linha)
            if m:
                ultimo = m
                break
        if not ultimo:
            return

        vals = [float(v) for v in ultimo.groups()]
        if all(v == 0 for v in vals):
            return

        neg = [0.0 if v == 0 else round(-v, 2) for v in vals]
        self.serial.send(f"G1 X{neg[0]:g} Y{neg[1]:g} Z{neg[2]:g} A{neg[3]:g} B{neg[4]:g} C{neg[5]:g} F800")
        self.serial.send("G92 X0 Y0 Z0 A0 B0 C0")
        self.serial.send("G1 X0 Y0 Z0 A0 B0 C0 F800")

    def rotina_lapis_suporte(self):
        """Máquina de estados que pega o lápis da mesa e o encaixa no suporte."""
        if self.modo_juntas:
            print("Modo juntas ativo — use o Home para retornar antes de executar a rotina.")
            return

        b = self.base_offset
        P_lapis = np.array([-300, 210, 0]) - b
        P_apr_lapis = P_lapis + np.array([0, 0, 100])

        if self.ponto_alcancavel(P_lapis):
            print("alcancavel")
        else:
            print("nao alcancavel")
        
        P_suporte = np.array([10, 120, 250]) - b
        P_apr_suporte = P_suporte + np.array([0, 0, 100])

        self.R_baixo = np.array([[ 0,  -1,  0], 
                                 [ -1, 0,  0], 
                                 [ 0,  0, -1]]) 
        
        alpha = np.rad2deg(np.arctan2(abs(P_suporte[1] - b[1]), abs(P_suporte[0] - b[0])))
        gama = -(180 - alpha) if b[0] > P_suporte[0] else -alpha
        gama_rad = np.deg2rad(gama)

        R_vertical = np.array([[-np.sin(gama_rad), 0, np.cos(gama_rad)],
                               [np.cos(gama_rad), 0, np.sin(gama_rad)],
                               [0,                1, 0]], dtype=float)

        # Estado 1: Home -> P_apr_lapis (Bézier)
        x, y, z = bz.calculo_pontos(self.P0, P_apr_lapis, self.Ri, self.R_baixo)
        A, B, C = self.interpolar_abc(self.Ri, self.P0, self.R_baixo, P_lapis, 21)
        self._mover_e_aguardar(x, y, z, A, B, C)
        self.Ri = self.R_baixo
        self.P0 = P_apr_lapis

        # Estado 2: Descer, pegar lápis e recuar (Linear)
        x, y, z = bz.calculo_linear(P_apr_lapis, P_lapis, self.R_baixo)
        A, B, C = np.full(21, A[-1]), np.full(21, B[-1]), np.full(21, C[-1])
        self._mover_e_aguardar(x, y, z, A, B, C)
        self.serial.send("M97 B0 T0.2") # Fecha a garra
        time.sleep(1)

        x, y, z = bz.calculo_linear(P_lapis, P_apr_lapis, self.R_baixo)
        self._mover_e_aguardar(x, y, z, A, B, C)
        self.P0 = P_apr_lapis

        # Estado 3: P_apr_lapis -> P_apr_suporte (Bézier)
        x, y, z = bz.calculo_pontos(self.P0, P_apr_suporte, self.Ri, R_vertical)
        A, B, C = self.interpolar_abc(self.Ri, self.P0, R_vertical, P_suporte, 21)
        self._mover_e_aguardar(x, y, z, A, B, C)
        self.Ri = R_vertical
        self.P0 = P_apr_suporte

        # Estado 4: Descer no suporte, soltar e recuar (Linear)
        x, y, z = bz.calculo_linear(P_apr_suporte, P_suporte, R_vertical)
        A, B, C = np.full(21, A[-1]), np.full(21, B[-1]), np.full(21, C[-1])
        self._mover_e_aguardar(x, y, z, A, B, C)
        self.serial.send("M97 B60 T0.2") # Abre a garra
        time.sleep(1)

        x, y, z = bz.calculo_linear(P_suporte, P_apr_suporte, R_vertical)
        self._mover_e_aguardar(x, y, z, A, B, C)
        self.P0 = P_apr_suporte

        # Estado 5: Voltar para Home
        self.home()

    def rotina_captura_calibracao(self, cap):
        b = self.base_offset
        c = np.array([-300, 210, 0]) - b
        pontos = calc_semi_esfera(c, r=500)

        for i, P_atual in enumerate(pontos):
            try:
                x, y, z = ik.calculo_angulos(P_atual[0], P_atual[1], P_atual[2])
                if not (-100 <= z <= 50):
                    continue
            except Exception:
                continue

            Z_e = c - P_atual
            Z_hat = Z_e / np.linalg.norm(Z_e)

            up = np.array([0, 0, 1])
            Y_e = up - np.dot(up, Z_hat) * Z_hat
            if np.linalg.norm(Y_e) < 1e-4:
                up = np.array([0, 1, 0])
                Y_e = up - np.dot(up, Z_hat) * Z_hat
            Y_hat = Y_e / np.linalg.norm(Y_e)

            X_hat = np.cross(Y_hat, Z_hat)
            X_hat = X_hat / np.linalg.norm(X_hat)

            R_matrix = np.column_stack((X_hat, Y_hat, Z_hat))
            try:
                A, B, C = ik.calculo_angulos_abc(R_matrix, P_atual, compensar_de=False)
            except Exception:
                continue
            C = 0
            self.enviar_juntas(x, y, z, A, B, C)
            if i == 0:
                time.sleep(15)
            else:
                delta_graus = np.max(np.abs(np.array([x, y, z]) - np.array([x_ant, y_ant, z_ant])))
                tempo_espera = max(4.0, (delta_graus / 800.0) * 60 * 1.2)
                time.sleep(tempo_espera)
            x_ant, y_ant, z_ant = x, y, z
            
            print("Realizando a captura de imagem da calibração...")
            ret, frame = cap.read()
            if ret:
                os.makedirs("capturas", exist_ok=True)
                P_mesa = P_atual + self.base_offset
                nome_arquivo = f"pos(x={P_mesa[0]:.2f}__y={P_mesa[1]:.2f}__z={P_mesa[2]:.2f}).jpg"
                caminho_arquivo = os.path.join("capturas", nome_arquivo)
                sucesso = cv2.imwrite(caminho_arquivo, frame)
                if sucesso:
                    print(f"Imagem salva com sucesso como {caminho_arquivo}")
                else:
                    print(f"Erro: OpenCV falhou ao salvar {caminho_arquivo}")
            else:
                print("Erro: Falha ao ler o frame da câmera já aberta.")
            time.sleep(2)

        self.enviar_juntas(0, 0, 0, 0, 0, 0)

    def rotina_manipular(self, p0, p3, Ri, Ac, Bc, Cc):
        if not self.ponto_alcancavel(p3 - self.base_offset, self.R_baixo):
            print("Objeto não alcançavel")
            return
        else:
            p3 -= self.base_offset
            x, y, z = bz.calculo_pontos(p0, p3, Ri, self.R_baixo)
            A, B, C = self.interpolar_abc(Ri, p0, self.R_baixo, p3, 21, Ac, Bc, Cc)
            self._mover_e_aguardar(x, y, z, A, B, C)
            self.P0 = p3
            self.Ri = self.R_baixo
            x, y, z = self.home()
            time.sleep(15)
            self.serial.send("M97 B60 T0.2")