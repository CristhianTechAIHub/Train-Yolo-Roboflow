# 👁️ YOLO Real-Time Detection

Detección de objetos en tiempo real usando **YOLOv8** y **OpenCV**. Captura video desde la cámara web y realiza inferencia frame a frame con visualización de bounding boxes.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-purple) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

---

## Estructura del proyecto

```
TUTORIAL_YOLO/
├── model/
│   └── best.pt                   # Modelo entrenado (PyTorch)
├── main.py                       # Script principal de inferencia
├── webServerApiSettings.json     # Configuración del servidor
└── YOLO_Training_Colab.ipynb     # Notebook de entrenamiento
```

---

## Requisitos

- Python 3.8+
- `ultralytics` — motor de inferencia YOLO
- `opencv-python` — captura y visualización de video
- Cámara web (resolución mínima recomendada: 720p)

---

## Instalación

```bash
# Instalar dependencias
pip install ultralytics opencv-python

# Colocar el modelo entrenado en la carpeta correcta
# model/best.pt  ← tu modelo exportado desde Colab
```

---

## Uso

```bash
python main.py
```

Presiona `q` para cerrar el stream.

---

## Configuración

Edita las constantes al inicio de `main.py` según tus necesidades:

| Parámetro        | Valor por defecto | Descripción                          |
|------------------|-------------------|--------------------------------------|
| `MODEL_PATH`     | `model/best.pt`   | Ruta al modelo entrenado             |
| `CONF_THRESHOLD` | `0.9`             | Confianza mínima para una detección  |
| `IOU_THRESHOLD`  | `0.85`            | Umbral IoU para NMS                  |
| `CAMERA_INDEX`   | `0`               | Índice de la cámara (0 = predeterminada) |

---

## Cómo funciona

1. **Carga el modelo** — Lee `model/best.pt` con Ultralytics y lista las clases detectables.
2. **Captura video** — Abre la cámara a 1280×720 y lee frames en un loop continuo.
3. **Inferencia por frame** — Aplica YOLO a cada frame con los umbrales configurados y dibuja bounding boxes.
4. **Muestra resultado** — Renderiza el frame anotado con el conteo de detecciones en pantalla.

---

## Entrenamiento del modelo

El archivo `YOLO_Training_Colab.ipynb` contiene el flujo completo para entrenar el modelo con tu propio dataset en Google Colab. Al finalizar el entrenamiento, descarga el archivo `best.pt` generado y colócalo en la carpeta `model/`.

---

## Licencia

MIT
