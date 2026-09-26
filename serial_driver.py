import os
import serial
import time
from config import GRBL_SETTINGS, GCODE_LOG

class SerialDriver:
    def __init__(self, port='COM3', baudrate=115200):
        """Cria a conexão serial real ou simulada e abre o arquivo de log de G-Code."""
        self.ser = serial.Serial(port, baudrate, timeout=1)

        time.sleep(2)  # espera GRBL iniciar

        while self.ser.in_waiting:
            line = self.ser.readline().decode().strip()
            if line:
                print("BOOT:", line)
        
        # simular = os.environ.get("SIMULAR_GRBL", "1") == "1"
        # if simular:
        #     from mock_serial import MockSerialGRBL
        #     self.ser = MockSerialGRBL()
        # else:
        #     self.ser = serial.Serial(port, baudrate, timeout=1)

        self._log = open(GCODE_LOG, "a")

    def send(self, cmd):
        """Envia um comando G-Code, grava no log e aguarda a resposta 'ok' do GRBL."""
        print(f">> {cmd}")
        self._log.write(cmd + "\n")
        self._log.flush()
        self.ser.write((cmd + '\n').encode())
        while True:
            response = self.ser.readline().decode().strip()
            if response:
                print("<<", response)
            if response == 'ok':
                break

    def send_settings(self):
        """Envia os parâmetros de configuração ($) do GRBL para a placa."""
        for key, value in GRBL_SETTINGS.items():
            self.send(f"${key} = {value}")

    def reset_log(self):
        """Apaga o log, mantendo o arquivo vazio para a sessão atual."""
        self._log.seek(0)
        self._log.truncate()

    def close(self):
        """Fecha a conexão serial e o arquivo de log."""
        if self.ser:
            self.ser.close()
        self._log.close()