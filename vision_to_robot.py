import os

import cv2
import numpy as np

# Deslocamento entre o frame de MUNDO (mesa) e o frame da BASE do robo.
#   P_base = P_mundo - BASE_OFFSET
# Este e o mesmo vetor `b` que aparece em robot_control.mover_para() e na
# rotina de captura da calibracao. A cinematica (e portanto a mao-olho)
# trabalha na BASE; ja robot_control.mover_para() recebe MUNDO. Converter
# entre os dois e obrigatorio, senao todo ponto sai errado por ~660 mm.
BASE_OFFSET = np.array([-137.0, 645.0, 25.0])


class VisionToRobot:
    """
    Etapa 3 do pipeline de calibracao: converte deteccoes 2D (pixel) da CNN
    em vetores 3D (mm), no referencial da BASE do manipulador.

    Carrega as duas calibracoes ja feitas:
      - intrinseca (etapa 1): K, dist
      - mao-olho  (etapa 2): T_cam_flange (fixo, camera -> flange)
    """

    def __init__(self, intrinsic_path, hand_eye_path):

        if not os.path.exists(intrinsic_path):
            raise FileNotFoundError(
                f"Calibracao intrinseca nao encontrada: {intrinsic_path}"
            )
        if not os.path.exists(hand_eye_path):
            raise FileNotFoundError(
                f"Calibracao mao-olho nao encontrada: {hand_eye_path}"
            )

        intrinseca = np.load(intrinsic_path)
        self.K = intrinseca["K"]
        self.dist = intrinseca["dist"]

        mao_olho = np.load(hand_eye_path)
        self.T_cam_flange = mao_olho["T_cam_flange"]  # 4x4, camera -> flange

        self.fx, self.fy = self.K[0, 0], self.K[1, 1]
        self.cx, self.cy = self.K[0, 2], self.K[1, 2]

        # matriz "nova" usada para desdistorcer os frames - precisa ser
        # a MESMA usada no undistort_frame(), senao K e frame ficam
        # inconsistentes entre si
        self._novo_K = None

    def preparar_undistort(self, largura, altura):
        """Calcula o novo K (para alpha=0) uma unica vez, para um dado
        tamanho de frame. Chame isso antes do primeiro undistort_frame()."""

        self._novo_K, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, (largura, altura), alpha=0
        )

        # a partir de agora, pixel_para_ponto_camera deve usar o NOVO K,
        # ja que e ele que descreve a geometria do frame ja desdistorcido
        self.fx, self.fy = self._novo_K[0, 0], self._novo_K[1, 1]
        self.cx, self.cy = self._novo_K[0, 2], self._novo_K[1, 2]

    def undistort_frame(self, frame):
        """Remove a distorcao de lente do frame (BGR). Chame isso ANTES
        de passar o frame para a YOLO."""

        h, w = frame.shape[:2]

        if self._novo_K is None:
            self.preparar_undistort(w, h)

        mapx, mapy = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self._novo_K, (w, h), cv2.CV_32FC1
        )

        return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    def pixel_para_ponto_camera(self, u, v, Z_mm):
        """Projecao inversa do modelo pinhole: pixel + profundidade -> ponto
        3D no referencial da CAMERA (em mm)."""

        X = (u - self.cx) * Z_mm / self.fx
        Y = (v - self.cy) * Z_mm / self.fy

        return np.array([X, Y, Z_mm, 1.0])

    def ponto_camera_para_base(self, ponto_cam_h, R_flange_base, t_flange_base):
        """Transforma um ponto do frame da camera para o frame da base do
        robo: camera -> flange -> base."""

        T_flange_base = np.eye(4)
        T_flange_base[:3, :3] = R_flange_base
        T_flange_base[:3, 3] = t_flange_base

        T_cam_base = T_flange_base @ self.T_cam_flange
        ponto_base_h = T_cam_base @ ponto_cam_h

        return ponto_base_h[:3]  # (X, Y, Z) em mm, referencial da base

    def vetor_para_deteccao(self, u, v, Z_camera_mm, R_flange_base, t_flange_base):
        """OBSOLETO - prefira pixel_para_base_no_plano() / deteccao_para_mundo().

        Este modo supoe que TODOS os objetos estao a uma mesma distancia
        Z_camera_mm ao longo do eixo optico. Isso so vale para o pixel exato
        do centro da imagem: como a mesa e um plano e nao uma esfera, a
        distancia ate ela cresce conforme o pixel se afasta do centro, e o
        erro chega facilmente a dezenas de mm nas bordas.

        Alem disso devolve o ponto no frame da BASE, enquanto
        robot_control.mover_para() espera o frame de MUNDO.

        Mantido so para nao quebrar codigo antigo.
        """

        ponto_cam = self.pixel_para_ponto_camera(u, v, Z_camera_mm)

        return self.ponto_camera_para_base(ponto_cam, R_flange_base, t_flange_base)

    # ------------------------------------------------------------------
    # Conversao pixel -> mm por INTERSECAO RAIO-PLANO
    #
    # E o metodo correto quando os objetos estao sobre uma superficie
    # plana conhecida (a mesa). Nao precisa de sensor de profundidade nem
    # de "chutar" um Z: o pixel define um raio que sai do centro optico,
    # e o ponto procurado e onde esse raio fura o plano da mesa.
    # ------------------------------------------------------------------

    def direcao_do_pixel(self, u, v, frame_desdistorcido=False):
        """Pixel -> vetor direcao (nao normalizado) no referencial da CAMERA.

        frame_desdistorcido=False (padrao): o pixel veio de um frame BRUTO,
        entao a distorcao da lente e removida aqui, no ponto. E o caso da
        DetectionThread quando ela roda sem vision_to_robot.

        frame_desdistorcido=True: o frame ja passou por undistort_frame(),
        entao usamos o novo K e nao aplicamos a distorcao de novo.
        """
        if frame_desdistorcido:
            if self._novo_K is None:
                raise RuntimeError(
                    "preparar_undistort() precisa ser chamado antes de usar "
                    "frame_desdistorcido=True."
                )
            x = (float(u) - self._novo_K[0, 2]) / self._novo_K[0, 0]
            y = (float(v) - self._novo_K[1, 2]) / self._novo_K[1, 1]
            return np.array([x, y, 1.0])

        # undistortPoints ja devolve coordenadas normalizadas (x', y'),
        # ou seja, o pixel corrigido e dividido por K de uma vez so.
        ponto = np.array([[[float(u), float(v)]]], dtype=np.float64)
        normalizado = cv2.undistortPoints(ponto, self.K, self.dist)
        x, y = normalizado[0, 0]
        return np.array([float(x), float(y), 1.0])

    def raio_na_base(self, u, v, T_flange_base, frame_desdistorcido=False):
        """Pixel -> raio (origem, direcao) no referencial da BASE do robo.

        A origem e o centro optico da camera; a direcao aponta para o objeto.
        """
        T_cam_base = np.asarray(T_flange_base) @ self.T_cam_flange

        origem = T_cam_base[:3, 3]
        direcao = T_cam_base[:3, :3] @ self.direcao_do_pixel(
            u, v, frame_desdistorcido
        )

        return origem, direcao

    def pixel_para_base_no_plano(self, u, v, T_flange_base, z_plano_base,
                                 frame_desdistorcido=False):
        """Pixel -> (X, Y, Z) em mm no referencial da BASE, sobre o plano
        horizontal z = z_plano_base.

        z_plano_base e a altura da superficie onde o objeto esta apoiado,
        JA no frame da base (mundo menos BASE_OFFSET).
        """
        origem, direcao = self.raio_na_base(
            u, v, T_flange_base, frame_desdistorcido
        )

        if abs(direcao[2]) < 1e-9:
            raise ValueError(
                "O raio da camera e paralelo ao plano da mesa: com esta pose "
                "de observacao o pixel nao define um ponto. Incline a camera "
                "na direcao da mesa."
            )

        s = (z_plano_base - origem[2]) / direcao[2]

        if s <= 0:
            raise ValueError(
                "O plano da mesa esta ATRAS da camera nesta pose "
                "(s = {:.1f}). Verifique a pose de observacao e o "
                "z_plano_base.".format(s)
            )

        return origem + s * direcao

    # ------------------------------------------------------------------
    # Conversao entre frames
    # ------------------------------------------------------------------

    @staticmethod
    def base_para_mundo(ponto_base):
        """Base -> mundo. E o frame que robot_control.mover_para() espera."""
        return np.asarray(ponto_base, dtype=float) + BASE_OFFSET

    @staticmethod
    def mundo_para_base(ponto_mundo):
        """Mundo -> base. E o frame em que a cinematica e a mao-olho vivem."""
        return np.asarray(ponto_mundo, dtype=float) - BASE_OFFSET

    def deteccao_para_mundo(self, bbox, T_flange_base, z_plano_mundo,
                            altura_objeto_mm=0.0, frame_desdistorcido=False):
        """Caminho completo: bbox da CNN -> (X, Y, Z) em mm no frame de MUNDO,
        pronto para entrar direto em robot_control.mover_para().

        bbox e [x1, y1, x2, y2] em pixels, como o YOLODetector devolve.
        Usamos o centro da caixa como ponto do objeto.

        z_plano_mundo e a altura da superficie de apoio no frame de MUNDO
        (a mesa e z = 0 se o tabuleiro da calibracao estava apoiado nela).

        altura_objeto_mm corrige o PARALAXE. O centro da bbox e o centroide
        visual do objeto, que fica a ~metade da altura dele, nao no chao.
        Se cruzarmos esse raio direto com o plano da mesa, o ponto sai
        deslocado para LONGE do eixo da camera, por

            erro = r * (h/2) / (H - h/2)

        com r = distancia horizontal ate o eixo da camera e H = altura da
        camera. Para h = 50 mm e r = 300 mm isso da ~15 mm — a mesma ordem
        do erro da propria calibracao. Cruzando o raio no plano da
        meia-altura o deslocamento some.

        O Z devolvido continua sendo o da BASE do objeto (z_plano_mundo),
        ou seja, onde ele encosta na mesa; a rotina de pega decide de que
        altura descer.
        """
        x1, y1, x2, y2 = bbox
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0

        altura_do_centroide = float(z_plano_mundo) + float(altura_objeto_mm) / 2.0

        ponto_base = self.pixel_para_base_no_plano(
            u, v,
            T_flange_base,
            z_plano_base=altura_do_centroide - BASE_OFFSET[2],
            frame_desdistorcido=frame_desdistorcido,
        )

        ponto_mundo = self.base_para_mundo(ponto_base)

        # X e Y vem do plano da meia-altura; o Z volta a ser o da mesa.
        ponto_mundo[2] = float(z_plano_mundo)

        return ponto_mundo
