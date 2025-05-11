import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from data_util import generate_synthetic_sequence_data, split_data
from model import create_ff_model, compile_model, train_model, evaluate_model, predict_sequence

# --- Configuración de la página ---
st.set_page_config(page_title="Predicción de Secuencias Cortas", layout="wide")
st.title("🔢 Patrones Numéricos")
st.markdown("Un modelo feedforward simple para predecir el siguiente valor (la suma) de una secuencia numérica.")

# --- Colores primarios y secundarios para el tema ---
primary_color = "#4c6ef5"  # Azul suave (énfasis)
secondary_color = "#a3aed0" # Gris azulado suave (secundario)
background_gradient = "linear-gradient(135deg, #e0f7fa 0%, #c5cae9 100%)" # Degradado suave azul a lila
text_color = "#37474f"      # Gris oscuro azulado

st.markdown(
    f"""
    <style>
        .stApp {{
            background: {background_gradient};
            color: {text_color};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {primary_color}; /* Títulos en color primario */
            font-weight: bold;
        }}
        p, div, stButton > button, stSlider > div > div > div > p {{
            color: {text_color};
            font-size: 16px;
        }}
        .stButton > button {{
            background-color: {primary_color};
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
        }}
        .stButton > button:hover {{
            background-color: {secondary_color};
            color: {text_color};
        }}
        .stTabs [data-baseweb="tab-list"] > div {{
            background-color: {secondary_color};
            border-bottom: 2px solid {primary_color};
        }}
        .stTabs [data-baseweb="tab-list"] > div > button[aria-selected="true"] {{
            background-color: {primary_color};
            color: white;
            font-weight: bold;
        }}
        .stProgress > div > div > div > div {{
            background-color: {primary_color};
        }}
        .stTextInput > label {{
            color: {primary_color};
            font-weight: bold;
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
    new_sequence_str = st.text_input(f"Secuencia de {st.session_state.get('sequence_length', 3)} números:", key="sequence_input")

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
        st.button("🗑️ Borrar Ingresado", on_click=clear_input)