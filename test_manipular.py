import numpy as np
import ik_craig as ik
from serial_driver import SerialDriver
from unity_client import UnityClient
from robot_control import RobotController

def testar():
    serial = SerialDriver()
    unity = UnityClient()
    controller = RobotController(serial, unity)

    # 1. Cinemática direta (mesmos parâmetros de gui.py)
    Rc, Pc = ik.cinematica_direta(-110, 0, -20, 0, -105, 0)
    Po = [-300, 210, 0]

    print(f"[FK] Pc calculado: {np.round(Pc, 2)}")
    p_offset = np.array(Po) - controller.base_offset
    alcancavel = controller.ponto_alcancavel(p_offset)
    print(f"[VOXEL] Ponto {p_offset} alcancavel: {alcancavel}")

    assert alcancavel, "Erro: Ponto fora do espaco de trabalho."

    # 2. Executa a rotina de manipulacao
    print("[EXEC] Executando rotina_manipular...")
    controller.rotina_manipular(Pc, Po, Rc, 0, -105, 0)
    print("[OK] Rotina manipular executada com sucesso.")

if __name__ == "__main__":
    testar()
