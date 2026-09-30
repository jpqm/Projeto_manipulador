import threading
import cv2
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap

from yolo_detector import YOLODetector
from detection_thread import DetectionThread

import ik_craig as ik
import numpy as np

import time

CAMERA_INDEX = 0
CAMERA_TICK_MS = 30


class CameraCanvas(QLabel):
    """Exibe o feed da webcam via OpenCV."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self._placeholder()

        self._cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            return

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._grab)
        self._timer.start(CAMERA_TICK_MS)

    def _placeholder(self):
        self.setText("Câmera indisponível")
        self.setStyleSheet(
            "background-color: #333; color: #aaa; font-size: 16px;"
        )

    def _grab(self):
        ok, frame = self._cap.read()
        if not ok:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        if self._cap:
            self._timer.stop()
            self._cap.release()
            self._cap = None
        super().closeEvent(event)


class RobotGUI(QMainWindow):
    """Janela principal do sistema de controle do robô."""
    update_status = pyqtSignal(str)

    def __init__(self, controller):
        """Monta a janela, conecta os sinais e inicia o timer de escuta do Unity."""
        super().__init__()

        self.controller = controller
        self.em_movimento = threading.Event()
        self.detector = YOLODetector()
        self.detection_thread = None

        self.update_status.connect(self.set_status)

        self.setWindowTitle("Controle do Manipulador com Gêmeo Digital")
        self.resize(800, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.timer_unity = QTimer(self)
        self.timer_unity.timeout.connect(self.verificar_botao_unity)
        self.timer_unity.start(100)

        left_panel = self._criar_painel_esquerdo()
        main_layout.addWidget(left_panel)

        self.camera = CameraCanvas(self)
        main_layout.addWidget(self.camera, 1)

    def _criar_painel_esquerdo(self) -> QWidget:
        """Monta o painel lateral com os controles e entradas numéricas."""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(280)

        group_pos = QGroupBox("Ângulos das Juntas")
        grid = QGridLayout()
        self.entradas = {}

        for i, nome in enumerate(["J1", "J2", "J3", "J4", "J5", "J6"]):
            lbl = QLabel(nome)
            lbl.setFixedWidth(30)
            entrada = QLineEdit("0")
            entrada.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, i, 0)
            grid.addWidget(entrada, i, 1)
            self.entradas[nome] = entrada

        group_pos.setLayout(grid)
        left_layout.addWidget(group_pos)

        self.status_lbl = QLabel("Aguardando comando...")
        self.status_lbl.setStyleSheet("color: gray; font-weight: bold;")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.status_lbl)

        btn_enviar = QPushButton("Enviar Juntas")
        btn_enviar.setMinimumHeight(40)
        btn_enviar.clicked.connect(self.enviar_pos)
        left_layout.addWidget(btn_enviar)

        btn_home = QPushButton("Ir para Home")
        btn_home.setMinimumHeight(30)
        btn_home.clicked.connect(self.home)
        left_layout.addWidget(btn_home)

        btn_rotina = QPushButton("Executar Tarefa Lápis")
        btn_rotina.setMinimumHeight(40)
        btn_rotina.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        btn_rotina.clicked.connect(self.executar_rotina_lapis)
        left_layout.addWidget(btn_rotina)

        btn_abrir = QPushButton("Abrir Garra")
        btn_abrir.clicked.connect(lambda: self.controller.serial.send("M97 B50 T0.2"))
        left_layout.addWidget(btn_abrir)

        btn_fechar = QPushButton("Fechar Garra")
        btn_fechar.clicked.connect(lambda: self.controller.serial.send("M97 B0 T0.2"))
        left_layout.addWidget(btn_fechar)

        btn_reconhecer = QPushButton("Reconhecer (YOLO)")
        btn_reconhecer.setMinimumHeight(40)
        btn_reconhecer.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        btn_reconhecer.clicked.connect(self.reconhecer)
        left_layout.addWidget(btn_reconhecer)

        btn_calibracao = QPushButton("Capturar calibração")
        btn_calibracao.setMinimumHeight(40)
        btn_calibracao.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        btn_calibracao.clicked.connect(self.executar_rotina_calibracao)
        left_layout.addWidget(btn_calibracao)

        log_group = QGroupBox("Log Deteccao")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #0f0; font-family: monospace; font-size: 11px;")
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(log_group)

        left_layout.addStretch()

        btn_encerrar = QPushButton("Encerrar Sistema")
        btn_encerrar.setStyleSheet("color: red;")
        btn_encerrar.clicked.connect(self.close)
        left_layout.addWidget(btn_encerrar)

        return left_panel

    # --- SLOTS E LÓGICA DE INTERACTION ---
    def set_status(self, text: str):
        self.status_lbl.setText(text)

    def verificar_botao_unity(self):
        unity = self.controller.unity
        serial = self.controller.serial

        if unity and unity.check_button_pressed():
            if self.controller.modo_juntas:
                self.set_status("Modo juntas ativo — pressione Home para retornar")
                return

            j1, j2, j3, j4, j5, j6 = unity.latest_angles

            print("=" * 50)
            print("[UNITY] Botão 'ENVIAR' pressionado na interface!")
            print(f"[UNITY] Ângulos lidos: J1:{j1} | J2:{j2} | J3:{j3} | J4:{j4} | J5:{j5} | J6:{j6}")

            self.controller.modo_juntas = True
            comando = f"G1 X{j1} Y{j2} Z{-j3} A{j4} B{-j6} C{j5} F800"
            if serial:
                serial.send(comando)
                print(f"[ARDUINO] Comando enviado: {comando}")
            print("=" * 50)

    def enviar_pos(self):
        if self.em_movimento.is_set():
            return
        try:
            valores = [float(self.entradas[nome].text()) for nome in ["J1", "J2", "J3", "J4", "J5", "J6"]]
        except ValueError:
            self.set_status("Erro: valores inválidos")
            return

        self.em_movimento.set()
        self.set_status("Enviando Juntas...")
        threading.Thread(target=self._executar_movimento, args=(valores,), daemon=True).start()

    def _executar_movimento(self, valores):
        self.controller.enviar_juntas(*valores)
        self.em_movimento.clear()
        self.update_status.emit("Comando de juntas enviado!")

    def home(self):
        def _go_home():
            self.em_movimento.set()
            self.update_status.emit("Retornando ao Home...")
            self.controller.home()
            self.em_movimento.clear()
            self.update_status.emit("Aguardando comando...")

        if not self.em_movimento.is_set():
            threading.Thread(target=_go_home, daemon=True).start()

    def executar_rotina_lapis(self):
        def _disparar():
            self.update_status.emit("Executando Pick & Place...")
            self.controller.rotina_lapis_suporte()
            self.update_status.emit("Rotina concluída!")

        if not self.em_movimento.is_set():
            threading.Thread(target=_disparar, daemon=True).start()

    def executar_rotina_calibracao(self):
        def _disparar():
            self.em_movimento.set()
            self.update_status.emit("Executando rotina de calibração...")
            try:
                self.controller.rotina_captura_calibracao(self.camera._cap)
                self.update_status.emit("Calibração concluída!")
            finally:
                self.em_movimento.clear()

        if not self.em_movimento.is_set():
            threading.Thread(target=_disparar, daemon=True).start()

    def reconhecer(self):
        if self.detection_thread and self.detection_thread.isRunning():
            self.set_status("Deteccao ja em andamento")
            return
        if not self.camera._cap or not self.camera._cap.isOpened():
            self.set_status("Camera indisponivel")
            return

        self.controller.enviar_juntas(-110, 0, -20, 0, -105, 0)

        time.sleep(15)

        self.camera._timer.stop()
        self.set_status("Detectando objetos...")
        self.log_text.append("Iniciando deteccao YOLO...")

        self.detection_thread = DetectionThread(self.camera._cap, self.detector)
        self.detection_thread.frame_ready.connect(self._on_frame_ready)
        self.detection_thread.detections_ready.connect(self._on_detections_ready)
        self.detection_thread.error_occurred.connect(self._on_detection_error)
        self.detection_thread.finished.connect(self._on_detection_finished)
        self.detection_thread.start()

    def _on_frame_ready(self, image):
        self.camera.setPixmap(QPixmap.fromImage(image))

    def _on_detections_ready(self, detections):
        if not detections:
            self.log_text.append("Nenhum objeto detectado")
            return
        for d in detections:
            msg = f"Detectado: {d['class']} ({d['confidence']:.2f})"
            self.log_text.append(msg)
            print(msg)
        self.set_status(f"{len(detections)} objeto(s) detectado(s)")

    def _on_detection_error(self, message):
        self.log_text.append(f"Erro: {message}")
        self.set_status("Erro na deteccao")

    def _on_detection_finished(self):
        self.set_status("Deteccao concluida")

    def manipular(self):
        Rc, Pc = ik.cinematica_direta(-110, 0, -20, 0, -105, 0)

        Po = [-300,210,0]

        self.controller.rotina_manipular(Pc, Po, Rc, 0, -105, 0)