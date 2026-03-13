# 🔍 Detección de Objetos con YOLOv5

Aplicación web para detección de objetos en tiempo real usando **YOLOv5** + **Streamlit**. Captura una imagen con tu cámara y el modelo identifica automáticamente los objetos presentes.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![YOLOv5](https://img.shields.io/badge/YOLO-v5su-green?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Demo

👉 [Ver aplicación en Streamlit Cloud](https://yolov5cmc.streamlit.app)

---

## ✨ Funcionalidades

- 📸 Captura de imagen directamente desde la cámara del dispositivo
- 🤖 Detección automática de **80 clases** de objetos (dataset COCO)
- 🎛️ Parámetros ajustables en tiempo real desde la barra lateral:
  - Umbral de confianza mínima
  - Umbral IoU (Non-Max Suppression)
  - Número máximo de detecciones
- 📊 Tabla resumen con categorías detectadas, cantidad y confianza promedio
- 📈 Gráfico de barras por categoría

---

## 🛠️ Tecnologías

| Tecnología | Uso |
|------------|-----|
| [Streamlit](https://streamlit.io) | Interfaz web |
| [Ultralytics YOLOv5](https://github.com/ultralytics/ultralytics) | Modelo de detección |
| [PyTorch](https://pytorch.org) | Backend de inferencia |
| [Pillow](https://python-pillow.org) | Procesamiento de imágenes |
| [NumPy](https://numpy.org) | Manipulación de arrays |
| [Pandas](https://pandas.pydata.org) | Tabla de resultados |

---

## 📁 Estructura del proyecto

```
yolov5/
├── app.py              # Aplicación principal
├── requirements.txt    # Dependencias Python (pip)
├── packages.txt        # Dependencias del sistema (apt-get)
└── README.md
```

---

## ⚙️ Instalación local

### Prerrequisitos
- Python 3.11+
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/cmcorrea4/yolov5.git
cd yolov5

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501`

---

## ☁️ Despliegue en Streamlit Cloud

1. Sube el repositorio a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. Selecciona `app.py` como archivo principal
5. Haz clic en **Deploy**

> **Nota:** Los archivos `requirements.txt` y `packages.txt` son detectados automáticamente por Streamlit Cloud para instalar dependencias Python y del sistema operativo respectivamente.

---

## 📦 Dependencias del sistema

El archivo `packages.txt` instala las librerías del sistema operativo necesarias para que OpenCV funcione en el servidor de Streamlit Cloud (Debian/Ubuntu):

```
libgl1
libglib2.0-0t64
```

---

## 🎯 Clases detectables

El modelo `yolov5su.pt` está pre-entrenado en el dataset **COCO** y detecta 80 clases:

<details>
<summary>Ver todas las clases</summary>

`person` · `bicycle` · `car` · `motorcycle` · `airplane` · `bus` · `train` · `truck` · `boat` · `traffic light` · `fire hydrant` · `stop sign` · `parking meter` · `bench` · `bird` · `cat` · `dog` · `horse` · `sheep` · `cow` · `elephant` · `bear` · `zebra` · `giraffe` · `backpack` · `umbrella` · `handbag` · `tie` · `suitcase` · `frisbee` · `skis` · `snowboard` · `sports ball` · `kite` · `baseball bat` · `baseball glove` · `skateboard` · `surfboard` · `tennis racket` · `bottle` · `wine glass` · `cup` · `fork` · `knife` · `spoon` · `bowl` · `banana` · `apple` · `sandwich` · `orange` · `broccoli` · `carrot` · `hot dog` · `pizza` · `donut` · `cake` · `chair` · `couch` · `potted plant` · `bed` · `dining table` · `toilet` · `tv` · `laptop` · `mouse` · `remote` · `keyboard` · `cell phone` · `microwave` · `oven` · `toaster` · `sink` · `refrigerator` · `book` · `clock` · `vase` · `scissors` · `teddy bear` · `hair drier` · `toothbrush`

</details>

---

## 🧑‍💻 Uso

1. Abre la aplicación
2. Ajusta los parámetros en la barra lateral según necesites
3. Haz clic en **"Capturar imagen"** para activar la cámara
4. Toma la foto
5. El modelo procesará la imagen y mostrará:
   - La imagen con los bounding boxes dibujados
   - Una tabla con los objetos detectados
   - Un gráfico de barras con el conteo por categoría

---

## 📄 Licencia

MIT License — libre para usar, modificar y distribuir.
