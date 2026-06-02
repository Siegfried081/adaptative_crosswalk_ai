"""Captura uma imagem da webcam para validar a configuração.

Aquece o sensor por 4 segundos antes de capturar, verifica o brilho médio
do frame e salva como test_capture.jpg.
"""

import time

import cv2


def configurar_camera(
    device_index: int = 0, largura: int = 640, altura: int = 480
) -> cv2.VideoCapture:
    """Abre a webcam com formato MJPEG e resolução definida."""
    cap = cv2.VideoCapture(device_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    return cap


def aquecer_sensor(cap: cv2.VideoCapture, duracao_segundos: float = 4.0) -> None:
    """Descarta frames continuamente para estabilizar exposição e balanço de branco."""
    inicio = time.time()
    while time.time() - inicio < duracao_segundos:
        cap.read()


def analisar_brilho(frame) -> str:
    """Retorna uma string indicando se o brilho do frame está OK."""
    brilho = frame.mean()
    if brilho > 250:
        return f"AVISO: frame saturado (brilho={brilho:.1f})"
    if brilho < 5:
        return f"AVISO: frame muito escuro (brilho={brilho:.1f})"
    return f"Brilho normal ({brilho:.1f})"


def main() -> int:
    cap = configurar_camera()
    if not cap.isOpened():
        print("ERRO: não consegui abrir a webcam")
        return 1

    print("Webcam aberta. Aquecendo o sensor (4 segundos)...")
    aquecer_sensor(cap)

    print("Capturando frame...")
    ret, frame = cap.read()

    if not ret or frame is None:
        print("ERRO: não consegui capturar frame válido")
        cap.release()
        return 1

    print(analisar_brilho(frame))
    cv2.imwrite("test_capture.jpg", frame)
    print(f"Frame salvo em test_capture.jpg: {frame.shape}")

    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
