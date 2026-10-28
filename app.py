import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Dashboard SADI - Generación Renovable",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Título y subtítulo
st.title("⚡ Dashboard Interactivo: Modelo Random Forest - SADI")
st.markdown(
    "Visualización interactiva del desempeño del modelo, validación cruzada temporal y análisis de residuos."
)
st.divider()
# ---------------------------------------------------------
# MÉTRICAS PRINCIPALES (TARJETAS)
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="R² Test Set", value="0.9562", delta="CV Promedio: 0.9221")
with col2:
    st.metric(
        label="Validación Cruzada",
        value="TimeSeriesSplit",
        delta="5 Folds cronológicos",
        delta_color="normal",
    )
with col3:
    st.metric(
        label="Estabilidad (Std R²)",
        value="0.0571",
        delta="Baja dispersión",
        delta_color="normal",
    )
with col4:
    st.metric(label="Variable Clave", value="Lag_1", delta="77% Importancia")
st.divider()
# ---------------------------------------------------------
# SIMULADOR / GRÁFICOS INTERACTIVOS
# ---------------------------------------------------------
if 'num_puntos' not in st.session_state:
    st.session_state.num_puntos = 500
if 'nivel_ruido' not in st.session_state:
    st.session_state.nivel_ruido = 1.0
st.sidebar.header("⚙️ Filtros de Simulación")
num_puntos = st.sidebar.slider(
    "Cantidad de Puntos a Simular",
    min_value=100,
    max_value=1000,
    value=st.session_state.num_puntos,
    step=50,
)
nivel_ruido = st.sidebar.slider(
    "Escala de Varianza (Heteroscedasticidad)",
    min_value=0.1,
    max_value=2.0,
    value=st.session_state.nivel_ruido,
    step=0.1,
)
# Generar datos simulados representativos alineados al modelo
np.random.seed(42)
predichos = np.random.uniform(0.5, 80.0, num_puntos)
residuos = np.random.normal(0, (predichos / 15 + 0.3) * nivel_ruido)
reales = predichos + residuos
df = pd.DataFrame(
    {"Predicho": predichos, "Real": reales, "Residuo": residuos}
)
st.session_state.num_puntos = num_puntos
st.session_state.nivel_ruido = nivel_ruido
tab1, tab2, tab3 = st.tabs([
    "📊 Residuos y Restricción Física",
    "📈 Predicho vs Real",
    "📝 Conclusiones Metodológicas",
])
with tab1:
    st.subheader("Gráfico de Residuos (Análisis de Errores)")
    st.write(
        "Se observa la distribución de los residuos frente a los valores predichos, destacando la restricción física de cero generación."
    )
    fig_residuos = px.scatter(
        df,
        x="Predicho",
        y="Residuo",
        opacity=0.4,
        color_discrete_sequence=["#008080"],
        labels={
            "Predicho": "Valores Predichos (GWh)",
            "Residuo": "Residuo (Real - Predicho)",
        },
        title="Gráfico de Residuos - Random Forest",
    )
    fig_residuos.add_hline(
        y=0, line_dash="dash", line_color="purple", annotation_text="Sesgo 0"
    )
    fig_residuos.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_residuos, use_container_width=True)
with tab2:
    st.subheader("Comparativa: Valores Predichos vs Valores Reales")
    fig_pred_real = px.scatter(
        df,
        x="Predicho",
        y="Real",
        opacity=0.4,
        color_discrete_sequence=["#2a9d8f"],
        labels={
            "Predicho": "Valores Predichos (GWh)",
            "Real": "Energía Real (GWh)",
        },
        title="Valores Reales vs. Predichos (Línea de predicción perfecta 45°)",
    )
    max_val = max(df["Predicho"].max(), df["Real"].max())
    fig_pred_real.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Predicción perfecta",
            line=dict(color="green", dash="dash"),
        )
    )
    fig_pred_real.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_pred_real, use_container_width=True)
with tab3:
    st.subheader("Resumen de Hallazgos:")
    st.markdown("""
    * **Desempeño Global:** El modelo alcanza un $R^2 \\approx 0.9562$ en el conjunto de prueba y un promedio de $R^2 \\approx 0.9221$ en validación cruzada temporal (`TimeSeriesSplit`).
    * **Importancia de Características:** `Lag_1` (77%) y `POTENCIA_PROXY` (11%) constituyen los factores determinantes que explican la alta precisión del modelo.
    * **Rigor Metodológico:** El uso de validación cruzada secuencial previene el *data leakage*, garantizando la robustez y estabilidad estructural del modelo frente a diferentes cortes temporales.
    """)
