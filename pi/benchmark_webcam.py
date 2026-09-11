"""Mede o FPS do YOLO sobre frames capturados da webcam.

Roda 100 frames consecutivos, mede o tempo de inferência de cada um e
salva imagens anotadas a cada 25 frames para inspeção visual.
"""

import time

import cv2
from ultralytics import YOLO

MODELO = "/home/mozardo/best480_ncnn_model"
IMGSZ = 480
N_FRAMES = 100
CONF = 0.25


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


def aquecer_modelo(model: YOLO, cap: cv2.VideoCapture, n_inferencias: int = 5) -> None:
    """Roda algumas inferências para descartar o overhead da primeira execução."""
    for _ in range(n_inferencias):
        ret, frame = cap.read()
        if ret:
            model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)


def calcular_estatisticas(tempos: list) -> dict:
    tempo_medio = sum(tempos) / len(tempos)
    return {
        "tempo_medio_ms": tempo_medio * 1000,
        "fps_medio": 1 / tempo_medio,
        "fps_min": 1 / max(tempos),
        "fps_max": 1 / min(tempos),
    }


def main() -> int:
    print(f"Carregando modelo: {MODELO}")
    model = YOLO(MODELO, task="detect")
    print(f"Classes: {model.names}")

    cap = configurar_camera()
    if not cap.isOpened():
        print("ERRO: não consegui abrir a webcam")
        return 1

    print("Aquecendo o sensor (4 segundos)...")
    aquecer_sensor(cap)

    print("Aquecendo o modelo...")
    aquecer_modelo(model, cap)

    print(f"Medindo {N_FRAMES} frames a {IMGSZ}px...")
    tempos = []
    total_deteccoes = 0
    frame_count = 0

    while frame_count < N_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar frame")
            continue

        t0 = time.perf_counter()
        results = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)
        dt = time.perf_counter() - t0

        tempos.append(dt)
        n_det = len(results[0].boxes)
        total_deteccoes += n_det

        if frame_count % 25 == 0:
            annotated = results[0].plot()
            cv2.imwrite(f"detect_{frame_count}.jpg", annotated)
            print(f"  Frame {frame_count}: {n_det} detecção(ões), {dt*1000:.1f} ms")

        frame_count += 1

    cap.release()

    stats = calcular_estatisticas(tempos)
    print("\n=== RESULTADO DO BENCHMARK ===")
    print(f"Modelo:                  {MODELO}")
    print(f"Resolução de inferência: {IMGSZ}px")
    print(f"Frames medidos:          {len(tempos)}")
    print(f"Tempo médio:             {stats['tempo_medio_ms']:.1f} ms")
    print(f"FPS médio:               {stats['fps_medio']:.2f}")
    print(f"FPS mínimo:              {stats['fps_min']:.2f}")
    print(f"FPS máximo:              {stats['fps_max']:.2f}")
    print(f"Total de detecções:      {total_deteccoes}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
