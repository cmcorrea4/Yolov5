import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os

st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")  # Descarga automática
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara.")

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_model()

if model:
    with st.sidebar:
        st.title("Parámetros")
        st.subheader("Configuración de detección")
        conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
        iou_threshold  = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
        max_det        = st.number_input("Detecciones máximas", 10, 2000, 1000, 10)

    picture = st.camera_input("Capturar imagen", key="camera")

    if picture:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("Detectando objetos..."):
            try:
                results = model(
                    cv2_img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=max_det
                )
            except Exception as e:
                st.error(f"Error durante la detección: {str(e)}")
                st.stop()

        result     = results[0]
        boxes      = result.boxes
        annotated  = result.plot()  # imagen con bounding boxes dibujadas

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen con detecciones")
            st.image(annotated, channels="BGR", use_container_width=True)

        with col2:
            st.subheader("Objetos detectados")

            if boxes and len(boxes) > 0:
                label_names = model.names
                data = []

                # Agrupar por categoría
                category_count = {}
                category_conf  = {}

                for box in boxes:
                    cat  = int(box.cls.item())
                    conf = float(box.conf.item())
                    category_count[cat] = category_count.get(cat, 0) + 1
                    category_conf.setdefault(cat, []).append(conf)

                for cat, count in category_count.items():
                    data.append({
                        "Categoría":          label_names[cat],
                        "Cantidad":           count,
                        "Confianza promedio": f"{np.mean(category_conf[cat]):.2f}"
                    })

                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("Categoría")["Cantidad"])
            else:
                st.info("No se detectaron objetos con los parámetros actuales.")
                st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")
else:
    st.error("No se pudo cargar el modelo. Verifica las dependencias e inténtalo nuevamente.")
    st.stop()

st.markdown("---")
st.caption("**Acerca de la aplicación**: Detección de objetos con YOLOv5 + Streamlit + PyTorch.")
