"""Grava vídeo da webcam com bounding boxes do YOLO desenhadas.

Captura por um período definido (30 segundos por padrão), processa cada
frame com o YOLO e salva um arquivo MP4 com as detecções.
"""

import time

import cv2
from ultralytics import YOLO

MODELO = "/home/mozardo/best480_ncnn_model"
IMGSZ = 480
CONF = 0.25
DURACAO_SEG = 30
ARQUIVO_SAIDA = "deteccao.mp4"
FPS_SAIDA = 5


def configurar_camera(
    device_index: int = 0, largura: int = 640, altura: int = 480
) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    return cap


def aquecer_sensor(cap: cv2.VideoCapture, duracao_segundos: float = 4.0) -> None:
    inicio = time.time()
    while time.time() - inicio < duracao_segundos:
        cap.read()


def criar_gravador(
    cap: cv2.VideoCapture, arquivo_saida: str, fps_saida: int
) -> cv2.VideoWriter:
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(arquivo_saida, fourcc, fps_saida, (largura, altura))


def main() -> int:
    print(f"Carregando modelo: {MODELO}")
    model = YOLO(MODELO, task="detect")

    cap = configurar_camera()
    if not cap.isOpened():
        print("ERRO: não consegui abrir a webcam")
        return 1

    print("Aquecendo o sensor (4 segundos)...")
    aquecer_sensor(cap)

    out = criar_gravador(cap, ARQUIVO_SAIDA, FPS_SAIDA)

    print(f"Gravando por {DURACAO_SEG} segundos...")
    inicio = time.time()
    frame_count = 0

    while time.time() - inicio < DURACAO_SEG:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)
        frame_anotado = results[0].plot()
        out.write(frame_anotado)

        n_det = len(results[0].boxes)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"  Frame {frame_count}: {n_det} detecção(ões)")

    cap.release()
    out.release()

    print(f"\nVídeo salvo: {ARQUIVO_SAIDA}")
    print(f"Total de frames gravados: {frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
