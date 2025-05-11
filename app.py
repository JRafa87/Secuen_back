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

# --- Variables ---
MODEL_FILE = "sequence_predictor_model.h5"

# --- Estados de sesión ---
if 'model' not in st.session_state:
    st.session_state.model = None

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

if 'sequence_length' not in st.session_state:  # ✅ MODIFICADO
    st.session_state.sequence_length = 3

# --- Intentar cargar modelo al inicio ---
if os.path.exists(MODEL_FILE) and st.session_state.model is None:
    try:
        st.session_state.model = tf.keras.models.load_model(MODEL_FILE)
        st.session_state.model_trained = True
        st.sidebar.success("✅ Modelo cargado previamente.")
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar el modelo guardado: {e}")

# --- Pestañas ---
tab_info, tab_config, tab_predict = st.tabs(["ℹ️ Información", "⚙️ Configuración & Entrenamiento", "🔮 Predicción"])

# --- ℹ️ Información ---
with tab_info:
    st.header("ℹ️ Información del Proyecto")
    st.markdown("""
    Este proyecto demuestra cómo una red neuronal feedforward simple puede aprender a predecir la suma de una secuencia corta de números.
    ...
    """)

# --- ⚙️ Configuración & Entrenamiento ---
with tab_config:
    st.header("⚙️ Configuración del Entrenamiento")

    sequence_length = st.slider("Longitud de la Secuencia:", 2, 5, value=3)
    num_sequences = st.slider("Número de Secuencias:", 100, 2000, value=1000, step=100)
    epochs = st.slider("Número de Épocas:", 10, 100, value=50, step=10)
    learning_rate = st.slider("Tasa de Aprendizaje:", 0.0001, 0.01, value=0.001, step=0.0001)

    if st.button("🚀 Entrenar Modelo"):
        with st.spinner(f"Entrenando el modelo durante {epochs} épocas..."):
            data, targets = generate_synthetic_sequence_data(num_sequences, sequence_length)
            train_data, train_targets, test_data, test_targets = split_data(data, targets)

            model = create_ff_model(sequence_length)
            model = compile_model(model, learning_rate=learning_rate)
            history = train_model(model, train_data, train_targets, epochs=epochs, verbose=0)

            st.session_state.model = model
            st.session_state.model_trained = True
            st.session_state.sequence_length = sequence_length  # ✅ GUARDAMOS sequence_length

            try:
                tf.keras.models.save_model(model, MODEL_FILE)
                st.success(f"💾 Modelo guardado como {MODEL_FILE}")
            except Exception as e:
                st.error(f"❌ Error al guardar el modelo: {e}")

            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Pérdida (Entrenamiento)')
            ax.plot(history.history['val_loss'], label='Pérdida (Validación)')
            ax.set_xlabel('Época')
            ax.set_ylabel('Error Cuadrático Medio (MSE)')
            ax.legend()
            st.pyplot(fig)

# --- 🔮 Predicción ---
with tab_predict:
    st.header("🔢 Predicción de Nueva Secuencia")
    st.markdown(f"Introduce una secuencia de **{st.session_state.sequence_length}** números separados por comas:")

    new_sequence_str = st.text_input("Secuencia:", key="sequence_input")

    def clear_input():
        st.session_state["sequence_input"] = ""

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("✨ Predecir"):
            if st.session_state.model_trained and st.session_state.model is not None:
                try:
                    new_sequence = [float(x.strip()) for x in new_sequence_str.split(',')]
                    if len(new_sequence) == st.session_state.sequence_length:  # ✅ Comprobamos contra la sesión
                        prediction = predict_sequence(st.session_state.model, new_sequence)
                        st.success(f"🔢 La predicción para {new_sequence} es: **{prediction:.2f}**")
                    else:
                        st.error(f"Por favor, introduce exactamente {st.session_state.sequence_length} números.")
                except ValueError:
                    st.error("❌ Por favor, introduce solo números válidos separados por comas.")
            else:
                st.warning("⚠️ Entrena el modelo primero en la pestaña 'Configuración & Entrenamiento'.")
    with col2:
        st.button("🗑️ Borrar Ingresado", on_click=clear_input)
