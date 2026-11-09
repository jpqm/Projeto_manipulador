import serial
import time
from config import GRBL_SETTINGS, GCODE_LOG

class SerialDriver:
    def __init__(self, port='COM3', baudrate=115200):
        """Cria a conexão serial real ou simulada e abre o arquivo de log de G-Code."""
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # espera GRBL iniciar
        except (serial.SerialException, OSError):
            from mock_serial import MockSerialGRBL
            print(f"[SerialDriver] Dispositivo na porta {port} não encontrado. Usando MockSerialGRBL.")
            self.ser = MockSerialGRBL(port=port, baudrate=baudrate)

        while self.ser.in_waiting:
            line = self.ser.readline().decode().strip()
            if line:
                print("BOOT:", line)

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

    def wait_idle(self, intervalo=0.1):
        """Consulta o status ('?') até o GRBL reportar Idle.
        Retorna False se não vier relatório de status ou o estado for inesperado (Alarm, Door...)."""
        while True:
            time.sleep(intervalo)  # ponytail: atraso fixo cobre a corrida 'ok' -> início do ciclo; aumentar se ler Idle cedo demais
            self.ser.write(b"?")   # comando em tempo real: sem '\n', senão o GRBL responde um 'ok' extra
            linha = self.ser.readline().decode(errors="ignore").strip()
            while linha and not linha.startswith("<"):  # descarta [MSG:...] e afins
                linha = self.ser.readline().decode(errors="ignore").strip()
            if linha.startswith("<Idle"):
                return True
            if not linha.startswith(("<Run", "<Jog", "<Hold", "<Home")):
                print(f"[SerialDriver] Status inesperado: {linha!r}")
                return False

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