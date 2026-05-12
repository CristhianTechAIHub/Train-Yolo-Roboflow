# main.py
import cv2
from ultralytics import YOLO

# ── CONFIGURACIÓN ─────────────────────────────────────────────
MODEL_PATH     = "model/best.pt"   # ← Ruta a tu modelo exportado
CONF_THRESHOLD = 0.9
IOU_THRESHOLD  = 0.85
CAMERA_INDEX   = 0                 # 0 = cámara por defecto
WINDOW_NAME    = "YOLO — Stream"

# ── CARGAR MODELO ─────────────────────────────────────────────
print(f"🤖 Cargando modelo: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"✅ Modelo listo | Clases: {model.names}")

# ── ABRIR CÁMARA ──────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"❌ No se pudo abrir la cámara (index={CAMERA_INDEX})")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("🎥 Stream iniciado — presiona 'q' para salir")

# ── LOOP DE INFERENCIA ────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️  No se pudo leer el frame")
        break

    # Inferencia
    results = model.predict(
        source  = frame,
        conf    = CONF_THRESHOLD,
        iou     = IOU_THRESHOLD,
        verbose = False,
    )

    # Dibujar bboxes sobre el frame
    annotated = results[0].plot()

    # Overlay: FPS + conteo de detecciones
    n_det = len(results[0].boxes)
    cv2.putText(annotated, f"Detecciones: {n_det}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, annotated)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Cerrando stream...")
        break

# ── LIBERAR RECURSOS ──────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()