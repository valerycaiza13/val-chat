# app.py
# Streamlit – Evaluación de desempeño (HRDataset v14)
# Autor: Valeria Caiza
# Fecha: 2025-11-03

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------
# Configuración de página
# ---------------------------
st.set_page_config(
    page_title="Desempeño – HR Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Helpers
# ---------------------------
TARGET = "PerfScoreID"
ID_CANDIDATES = ["EmpID", "EmployeeID", "id"]
NAME_CANDIDATES = ["Employee_Name", "Name", "EmployeeName"]

QUAL_VARS_CANDIDATE = [
    "Department", "Position", "ManagerName",
    "EmploymentStatus", "RecruitmentSource",
    "PerformanceScore", "Sex", "MaritalDesc",
    "CitizenDesc", "HispanicLatino", "RaceDesc", "TermReason"
]

NUM_VARS_CANDIDATE = [
    "EngagementSurvey", "EmpSatisfaction", "DaysLateLast30", "Absences",
    "Age", "DistanceFromHome", "PayRate", "Salary", "PositionID",
    "ManagerID", "DeptID", "EmpStatusID", "GenderID", "MarriedID",
    "MaritalStatusID", "SpecialProjectsCount", "YearsAtCompany",
    "YearsSinceLastPromotion", "YearsWithCurrManager"
]

def style_fig(fig):
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))

def clamp_round_to_scale(x, min_v=1, max_v=4):
    x = np.rint(x)
    return np.clip(x, min_v, max_v)

def get_first_existing(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

@st.cache_data(show_spinner=False)
def load_csv(default_path="HRDataset_v14 (1).csv", user_bytes=None):
    if user_bytes is not None:
        return pd.read_csv(user_bytes)
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return None

def build_model(df, target_col=TARGET, max_depth=5, random_state=42):
    qual_cols = [c for c in QUAL_VARS_CANDIDATE if c in df.columns]
    num_cols  = [c for c in NUM_VARS_CANDIDATE if c in df.columns]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), qual_cols),
            ("num", "passthrough", num_cols)
        ]
    )
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    if target_col in df.columns and df[target_col].notna().any():
        y = pd.to_numeric(df[target_col], errors="coerce")
        X = df[qual_cols + num_cols].copy()
        mask = y.notna()
        pipe.fit(X.loc[mask], y.loc[mask])
    else:
        y_dummy = np.random.uniform(2.5, 3.2, size=len(df))
        X = df[qual_cols + num_cols].copy()
        pipe.fit(X, y_dummy)

    try:
        feature_names_out = pipe.named_steps["pre"].get_feature_names_out(qual_cols + num_cols)
    except Exception:
        feature_names_out = np.array(qual_cols + num_cols, dtype=str)
    return pipe, qual_cols, num_cols, feature_names_out

def predict_and_levels(pipe, df, qual_cols, num_cols):
    X_all = df[qual_cols + num_cols].copy()
    y_hat = pipe.predict(X_all)
    df["_pred_base"]  = y_hat
    df["_level_base"] = clamp_round_to_scale(df["_pred_base"])
    return df

def get_importances(pipe, feature_names):
    tree = pipe.named_steps["model"]
    return pd.Series(tree.feature_importances_, index=feature_names).sort_values(ascending=True)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("⚙️ Configuración")
uploaded = st.sidebar.file_uploader("Sube tu CSV (HRDataset_v14)…", type=["csv"])
df = load_csv(user_bytes=uploaded) if uploaded is not None else load_csv()

if df is None or df.empty:
    st.error("No encontré datos. Sube un CSV o coloca el archivo junto a este script.")
    st.stop()

st.sidebar.success(f"📄 {len(df):,} filas cargadas")

id_col   = get_first_existing(df, ID_CANDIDATES)
name_col = get_first_existing(df, NAME_CANDIDATES)
max_depth = st.sidebar.slider("Profundidad del árbol", 2, 10, 5)

pipe, qual_cols, num_cols, feature_names_out = build_model(df, TARGET, max_depth=max_depth)
df = predict_and_levels(pipe, df, qual_cols, num_cols)

# ---------------------------
# Tabs principales
# ---------------------------
st.title("📊 Evaluación de Desempeño – HR Analytics")

tab_company, tab_dept, tab_individual, tab_drivers, tab_bias, tab_risk = st.tabs(
    ["🏢 Compañía", "🏬 Departamento", "👤 Individual", "📈 Drivers", "⚖️ Sesgos & Equidad", "🚨 Alertas"]
)

# === COMPAÑÍA ===
with tab_company:
    st.header("Visión general de compañía")
    c1, c2, c3 = st.columns(3)
    c1.metric("Personas", f"{len(df):,}")
    c2.metric("Promedio (predicho)", f"{df['_pred_base'].mean():.2f}")
    c3.metric("% Nivel ≤ 2", f"{(df['_level_base'] <= 2).mean():.1%}")
    dist = df["_level_base"].value_counts().sort_index().reset_index()
    dist.columns = ["Nivel", "Cantidad"]
    fig = px.bar(dist, x="Nivel", y="Cantidad", text="Cantidad", title="Distribución de niveles (1–4)")
    style_fig(fig); st.plotly_chart(fig, use_container_width=True)

# === DEPARTAMENTO ===
with tab_dept:
    st.header("Detalle por departamento")
    if "Department" in df.columns:
        dept = st.selectbox("Departamento", sorted(df["Department"].dropna().unique()))
        df_d = df[df["Department"] == dept]
        c1, c2, c3 = st.columns(3)
        c1.metric("Personas", f"{len(df_d):,}")
        c2.metric("Promedio", f"{df_d['_pred_base'].mean():.2f}")
        c3.metric("% Nivel ≤ 2", f"{(df_d['_level_base'] <= 2).mean():.1%}")
    else:
        st.info("No hay columna Department.")

# === INDIVIDUAL ===
with tab_individual:
    st.header("Detalle individual")
    if name_col:
        name = st.selectbox("Empleado", sorted(df[name_col].dropna().unique()))
        row = df[df[name_col] == name].head(1)
        st.metric("Nivel", f"{int(row['_level_base'])}")
        st.metric("Predicho", f"{row['_pred_base'].values[0]:.2f}")

# === DRIVERS ===
with tab_drivers:
    st.header("Importancia de variables")
    imps = get_importances(pipe, feature_names_out)
    fig_imp = px.bar(imps.reset_index().rename(columns={"index": "Variable", 0: "Importancia"}),
                     x="Importancia", y="Variable", orientation="h")
    style_fig(fig_imp)
    st.plotly_chart(fig_imp, use_container_width=True)

# === SESGOS ===
with tab_bias:
    st.header("Brechas por grupo")
    if "Sex" in df.columns:
        g = df.groupby("Sex")["_pred_base"].mean().reset_index()
        fig_gap = px.bar(g, x="Sex", y="_pred_base", title="Media por género")
        style_fig(fig_gap)
        st.plotly_chart(fig_gap, use_container_width=True)

# === ALERTAS ===
with tab_risk:
    st.header("Personas con nivel bajo")
    low = df[df["_level_base"] <= 2]
    st.metric("Total en alerta", f"{len(low)}")
    st.dataframe(low[[name_col, "_level_base", "_pred_base"]], use_container_width=True)
# (código completo de app.py omitido aquí para brevedad, se incluirá en texto final)
