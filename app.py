import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os  # Para verificar si el archivo del modelo existe

from data_util import generate_synthetic_sequence_data, split_data
from model import create_ff_model, compile_model, train_model, evaluate_model, predict_sequence

# --- Configuración de la página ---
st.set_page_config(page_title="Predicción de Secuencias Cortas", layout="wide")
st.title("🔢 Predicción de Secuencias Numéricas Cortas") # Cambié el emoji aquí
st.markdown("Un modelo feedforward simple para predecir el siguiente valor (la suma) de una secuencia numérica.")

# --- Colores primarios y secundarios para el tema ---
primary_color = "#673ab7"  # Morado oscuro
secondary_color = "#e91e63" # Rosa
background_color = "#f3e5f5" # Lila claro
text_color = "#212121"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {background_color};
            color: {text_color};
        }}
        h1, h2, h3, h4, h5, h6, p, div, stButton > button, stSlider > div > div > div > p {{
            color: {text_color};
        }}
        .stButton > button:hover {{
            background-color: {primary_color};
            color: white;
        }}
        .stTabs [data-baseweb="tab-list"] > div {{
            background-color: {primary_color};
        }}
        .stTabs [data-baseweb="tab-list"] > div > button[aria-selected="true"] {{
            background-color: {secondary_color};
            color: white;
        }}
        .stProgress > div > div > div > div {{
            background-color: {secondary_color};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Variables para el nombre del archivo del modelo ---
MODEL_FILE = "sequence_predictor_model.h5"

# --- Estado de la sesión para el modelo ---
if 'model' not in st.session_state:
    st.session_state.model = None

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# --- Cargar modelo al inicio si existe ---
if os.path.exists(MODEL_FILE) and st.session_state.model is None:
    try:
        st.session_state.model = tf.keras.models.load_model(MODEL_FILE)
        st.session_state.model_trained = True
        st.sidebar.success("✅ Modelo cargado previamente.")
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar el modelo guardado: {e}")

# --- Pestañas ---
tab_info, tab_config, tab_predict = st.tabs(["ℹ️ Información", "⚙️ Configuración & Entrenamiento", "🔮 Predicción"])

# --- Pestaña de Información ---
with tab_info:
    st.header("ℹ️ Información del Proyecto")
    st.markdown("""
    Este proyecto demuestra cómo una red neuronal feedforward simple puede aprender a predecir la suma de una secuencia corta de números.

    **¿Cómo funciona?**

    1.  **Generación de Datos:** Se generan secuencias numéricas aleatorias de una longitud configurable. El objetivo es predecir la suma de los números en cada secuencia.
    2.  **Modelo Feedforward:** Se utiliza una red neuronal con capas densas (fully connected). Cada número en la secuencia se trata como una característica de entrada.
    3.  **Entrenamiento:** La red aprende a mapear las secuencias a sus sumas utilizando el algoritmo de backpropagation y el optimizador Adam. La función de pérdida utilizada es el Error Cuadrático Medio (MSE).
    4.  **Predicción:** Una vez entrenado (o cargado), el modelo puede tomar una nueva secuencia de números y predecir su suma.

    **Uso:**

    1.  Ve a la pestaña "**⚙️ Configuración & Entrenamiento**" para ajustar los parámetros del entrenamiento y, si lo deseas, entrenar un nuevo modelo. Si ya existe un modelo guardado, se cargará automáticamente.
    2.  Ve a la pestaña "**🔮 Predicción**" e introduce una secuencia de números (separados por comas) en el cuadro de texto. Haz clic en "**✨ Predecir**" para obtener la predicción del modelo. Puedes usar el botón "**🗑️ Borrar Ingresado**" para limpiar el campo de entrada.

    **Nota:** Este es un ejemplo simplificado para ilustrar los conceptos básicos. Para problemas de predicción de series de tiempo más complejos, se suelen utilizar Redes Neuronales Recurrentes (RNNs).
    """)

# --- Pestaña de Configuración y Entrenamiento ---
with tab_config:
    st.header("⚙️ Configuración del Entrenamiento")
    st.markdown("Ajusta los parámetros para la generación y el entrenamiento del modelo.")

    sequence_length = st.slider("Longitud de la Secuencia:", min_value=2, max_value=5, value=3)
    num_sequences = st.slider("Número de Secuencias:", min_value=100, max_value=2000, value=1000, step=100)
    epochs = st.slider("Número de Épocas:", min_value=10, max_value=100, value=50, step=10)
    learning_rate = st.slider("Tasa de Aprendizaje:", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)

    # --- Botón de Entrenamiento ---
    if st.button("🚀 Entrenar Modelo"):
        with st.spinner(f"Entrenando el modelo durante {epochs} épocas..."):
            data, targets = generate_synthetic_sequence_data(num_sequences, sequence_length)
            train_data, train_targets, test_data, test_targets = split_data(data, targets)

            model = create_ff_model(sequence_length)
            model = compile_model(model, learning_rate=learning_rate)
            history = train_model(model, train_data, train_targets, epochs=epochs, verbose=0)
            st.session_state.model = model
            st.session_state.model_trained = True
            st.success("✅ ¡Modelo entrenado!")

            # Guardar el modelo después del entrenamiento
            try:
                tf.keras.models.save_model(st.session_state.model, MODEL_FILE)
                st.success(f"💾 Modelo guardado como {MODEL_FILE}")
            except Exception as e:
                st.error(f"❌ Error al guardar el modelo: {e}")

            # Mostrar la pérdida durante el entrenamiento
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Pérdida (Entrenamiento)')
            ax.plot(history.history['val_loss'], label='Pérdida (Validación)')
            ax.set_xlabel('Época')
            ax.set_ylabel('Error Cuadrático Medio (MSE)')
            ax.legend()
            st.pyplot(fig)

# --- Pestaña de Predicción ---
with tab_predict:
    st.header("🔮 Predicción de Nueva Secuencia")
    st.markdown(f"Introduce una secuencia numérica de {st.session_state.get('sequence_length', 3)} números (separados por comas) para predecir su suma.")
    new_sequence_str = st.text_input(f"Secuencia de {st.session_state.get('sequence_length', 3)} números:", key="sequence_input") # Agregué una key

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("✨ Predecir"):
            if st.session_state.model_trained and st.session_state.model is not None:
                try:
                    sequence_length_pred = st.session_state.get('sequence_length', 3)
                    new_sequence = [float(x.strip()) for x in new_sequence_str.split(',')]
                    if len(new_sequence) == sequence_length_pred:
                        prediction = predict_sequence(st.session_state.model, new_sequence)
                        st.subheader(f"Predicción para la secuencia: {new_sequence}")
                        st.success(f"El modelo predice: **{prediction:.2f}**")
                    else:
                        st.error(f"Por favor, introduce una secuencia de exactamente {sequence_length_pred} números.")
                except ValueError:
                    st.error("Por favor, introduce números válidos separados por comas.")
            else:
                st.info("Por favor, entrena el modelo primero en la pestaña de 'Configuración & Entrenamiento'.")
    with col2:
        if st.button("🗑️ Borrar Ingresado"):
            st.session_state["sequence_input"] = "" # Limpia el valor del text_input