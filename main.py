import sys
from PyQt6.QtWidgets import QApplication

from unity_client import UnityClient
from serial_driver import SerialDriver
from robot_control import RobotController
from gui import RobotGUI

def main():
    """Ponto de entrada: cria as dependências, recupera a posição anterior,
    inicia a interface PyQt e encerra o sistema com segurança."""
    # 1. Instancia as dependências do hardware e comunicação
    serial_drv = SerialDriver()
    unity_cli = UnityClient()
    controller = RobotController(serial_drv, unity_cli)

    # 2. Configurações iniciais do robô via G-Code
    serial_drv.send_settings()
    serial_drv.send("G90")
    serial_drv.send("F800")
    controller.recuperar_do_log()
    serial_drv.reset_log()
    serial_drv.send("M97 B40 T0.2")

    # 3. Inicia o ambiente PyQt
    app = QApplication(sys.argv)
    window = RobotGUI(controller)
    window.show()

    # 4. Inicia o loop da interface gráfica (vai travar aqui até o app ser fechado)
    exit_code = app.exec()

    # 5. Encerramento seguro ao fechar a janela
    serial_drv.close()
    unity_cli.close()
    print("Sistema encerrado com sucesso.")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()