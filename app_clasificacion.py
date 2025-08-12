# app_clasificacion.py
# Streamlit app: Clasificación binaria para insurance.csv
# Requisitos mínimos: streamlit, pandas, numpy, scikit-learn, matplotlib
# Ejecutar localmente: streamlit run app_clasificacion.py

from __future__ import annotations

import io
import textwrap
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# -------------------------
# Configuración de página
# -------------------------
st.set_page_config(
    page_title="Clasificación — insurance.csv",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🩺 Clasificación Binaria — Seguro Médico")
st.caption(
    "Crea la etiqueta `contrata_seguro` a partir de `charges` y entrena un modelo simple "
    "con preprocesamiento automático (one-hot + escalado)."
)


# -------------------------
# Utilidades
# -------------------------
@st.cache_data(show_spinner=False)
def load_csv(file, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    return pd.read_csv(file, sep=sep, encoding=encoding)


@st.cache_data(show_spinner=False)
def load_excel(file, sheet_name: Optional[str] = None) -> pd.DataFrame:
    return pd.read_excel(file, sheet_name=sheet_name)


def safe_coerce_numeric(series: pd.Series) -> pd.Series:
    """Intenta convertir a numérico, conservando NaN en errores."""
    return pd.to_numeric(series, errors="coerce")


def get_feature_types(df: pd.DataFrame, target_cols: List[str]) -> Tuple[List[str], List[str]]:
    """Devuelve listas de columnas numéricas y categóricas (excluyendo target_cols)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_cols = [c for c in numeric_cols if c not in target_cols]
    categorical_cols = [c for c in categorical_cols if c not in target_cols]

    return numeric_cols, categorical_cols


def impute_simple(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    """Imputación simple sobre una copia del DF para descargar versión 'limpia'."""
    out = df.copy()
    for c in num_cols:
        median_val = out[c].median(skipna=True)
        out[c] = out[c].fillna(median_val)
    for c in cat_cols:
        mode_val = out[c].mode(dropna=True)
        mode_val = mode_val.iloc[0] if len(mode_val) else "missing"
        out[c] = out[c].fillna(mode_val)
    return out


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("Real")
    ax.set_xlabel("Predicho")
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocessor


def probas_from_estimator(clf, X) -> Optional[np.ndarray]:
    """Obtiene proba de clase positiva si está disponible, o None."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        # Escalar a [0,1] vía min-max para simular probabilidad (no exacto, pero útil para ROC)
        smin, smax = np.min(scores), np.max(scores)
        if smax > smin:
            return (scores - smin) / (smax - smin)
        return None
    return None


# -------------------------
# Sidebar — Carga y opciones
# -------------------------
st.sidebar.header("⚙️ Configuración")

uploaded = st.sidebar.file_uploader(
    "Sube un archivo CSV/XLSX (o deja vacío para intentar 'insurance.csv' local)",
    type=["csv", "xlsx", "xls"],
)

filetype = st.sidebar.selectbox("Tipo de archivo", ["auto", "csv", "excel"], index=0)
sep = st.sidebar.text_input("Separador CSV", value=",")
encoding = st.sidebar.selectbox("Encoding", ["utf-8", "latin-1", "utf-16"], index=0)
sheet_name: Optional[str] = None

df: Optional[pd.DataFrame] = None

try:
    if uploaded is not None:
        if (filetype == "auto" and uploaded.name.lower().endswith((".xlsx", ".xls"))) or filetype == "excel":
            # Excel
            xls = pd.ExcelFile(uploaded)
            sheet_name = st.sidebar.selectbox("Hoja de Excel", xls.sheet_names, index=0)
            df = load_excel(uploaded, sheet_name=sheet_name)
        else:
            df = load_csv(uploaded, sep=sep, encoding=encoding)
    else:
        # fallback: intenta leer un insurance.csv local
        try:
            df = load_csv("insurance.csv", sep=",", encoding="utf-8")
            st.sidebar.info("Usando archivo local: insurance.csv")
        except Exception:
            df = None
except Exception as e:
    st.error(f"❌ Error al cargar el archivo: {e}")

if df is None:
    st.warning("Sube un archivo o coloca **insurance.csv** junto al script.")
    st.stop()

# -------------------------
# Validación de columnas
# -------------------------
if "charges" not in df.columns:
    st.error("El dataset no contiene la columna **'charges'** necesaria para crear la etiqueta binaria.")
    st.stop()

# Intento de coerce a numérico
df["charges"] = safe_coerce_numeric(df["charges"])

# Umbral para target
min_c = float(np.nanpercentile(df["charges"], 1))
max_c = float(np.nanpercentile(df["charges"], 99))
default_thr = 10000.0 if min_c <= 10000 <= max_c else float(np.nanmedian(df["charges"]))

threshold = st.sidebar.number_input(
    "Umbral para 'contrata_seguro' (charges > umbral)",
    min_value=float(np.floor(min_c)) if np.isfinite(min_c) else 0.0,
    max_value=float(np.ceil(max_c)) if np.isfinite(max_c) else 100000.0,
    value=float(default_thr),
    step=100.0,
)

# Crear target
df["contrata_seguro"] = (df["charges"] > threshold).astype(int)

# Mostrar resumen de datos
with st.expander("📦 Vista previa de datos", expanded=False):
    st.write("**Shape:**", df.shape)
    st.dataframe(df.head(10), use_container_width=True)
    st.write("**Columnas:**", list(df.columns))

# Distribución de la etiqueta
pos_rate = df["contrata_seguro"].mean()
st.markdown(f"**Distribución de clases** — 1s: {pos_rate:.2%} | 0s: {(1 - pos_rate):.2%}")

# -------------------------
# Selección de features y modelo
# -------------------------
drop_cols_default = ["contrata_seguro", "charges"]
numeric_cols_all, categorical_cols_all = get_feature_types(df, target_cols=drop_cols_default)

all_features = [c for c in df.columns if c not in drop_cols_default]
feat_selected = st.multiselect(
    "Selecciona features (excluye 'charges' y 'contrata_seguro')",
    options=all_features,
    default=all_features,
)

if not feat_selected:
    st.error("Selecciona al menos una feature para entrenar el modelo.")
    st.stop()

# Tipos de columnas en selección
num_cols = [c for c in numeric_cols_all if c in feat_selected]
cat_cols = [c for c in categorical_cols_all if c in feat_selected]

# Modelo
st.sidebar.subheader("🤖 Modelo")
model_name = st.sidebar.selectbox(
    "Algoritmo",
    ["LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier", "Dummy (estrategia 'most_frequent')"],
    index=0,
)

# Hiperparámetros básicos
params = {}
if model_name == "LogisticRegression":
    C = st.sidebar.number_input("C (inversa de regularización)", min_value=0.001, max_value=1000.0, value=1.0, step=0.1)
    max_iter = st.sidebar.number_input("max_iter", min_value=100, max_value=5000, value=1000, step=100)
    solver = st.sidebar.selectbox("solver", ["lbfgs", "liblinear", "saga"], index=0)
    params.update(dict(C=C, max_iter=int(max_iter), solver=solver))
elif model_name == "KNeighborsClassifier":
    n_neighbors = st.sidebar.slider("n_neighbors", min_value=1, max_value=50, value=5, step=1)
    weights = st.sidebar.selectbox("weights", ["uniform", "distance"], index=0)
    params.update(dict(n_neighbors=int(n_neighbors), weights=weights))
elif model_name == "DecisionTreeClassifier":
    max_depth = st.sidebar.slider("max_depth", min_value=1, max_value=50, value=5, step=1)
    min_samples_split = st.sidebar.slider("min_samples_split", min_value=2, max_value=20, value=2, step=1)
    params.update(dict(max_depth=int(max_depth), min_samples_split=int(min_samples_split)))
else:
    strategy = st.sidebar.selectbox("Estrategia dummy", ["most_frequent", "stratified"], index=0)
    params.update(dict(strategy=strategy))

# División holdout
st.sidebar.subheader("📐 Validación")
test_size = st.sidebar.slider("test_size (holdout)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
kfolds = st.sidebar.slider("K de StratifiedKFold (CV)", min_value=3, max_value=10, value=5, step=1)
random_state = st.sidebar.number_input("random_state", min_value=0, max_value=10_000, value=42, step=1)

# -------------------------
# Preparación de datos
# -------------------------
X = df[feat_selected].copy()
y = df["contrata_seguro"].astype(int).values

# Preprocesador
preprocessor = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols)

# Estimador
if model_name == "LogisticRegression":
    estimator = LogisticRegression(**params)
elif model_name == "KNeighborsClassifier":
    estimator = KNeighborsClassifier(**params)
elif model_name == "DecisionTreeClassifier":
    estimator = DecisionTreeClassifier(**params, random_state=random_state)
else:
    estimator = DummyClassifier(**params, random_state=random_state)

# Pipeline completo
pipe = Pipeline(steps=[("pre", preprocessor), ("clf", estimator)])

# -------------------------
# Holdout
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_proba = probas_from_estimator(pipe, X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

st.subheader("📊 Resultados — Holdout")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("Precision", f"{prec:.3f}")
col3.metric("Recall", f"{rec:.3f}")
col4.metric("F1", f"{f1:.3f}")
col5.metric("ROC-AUC", f"{roc_auc:.3f}" if np.isfinite(roc_auc) else "—")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
fig_cm = plot_confusion_matrix(cm, class_names=["0", "1"])
st.pyplot(fig_cm, use_container_width=True)

if y_proba is not None:
    fig_roc = plot_roc_curve(y_test, y_proba)
    st.pyplot(fig_roc, use_container_width=True)
else:
    st.info("El modelo no expone probabilidades; no se muestra ROC-AUC/curva ROC.")

# Reporte de clasificación
with st.expander("📃 Classification Report (holdout)"):
    report = classification_report(y_test, y_pred, digits=3, zero_division=0)
    st.code(report, language="text")

# -------------------------
# Validación Cruzada (CV)
# -------------------------
st.subheader("🔁 Validación Cruzada (StratifiedKFold)")

skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=random_state)
cv_accuracy = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
st.write(f"**Accuracy CV ({kfolds}-Fold)**: {cv_accuracy.mean():.3f} ± {cv_accuracy.std():.3f}")

# ROC-AUC en CV si es posible
try:
    cv_roc = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
    st.write(f"**ROC-AUC CV ({kfolds}-Fold)**: {cv_roc.mean():.3f} ± {cv_roc.std():.3f}")
except Exception as e:
    st.info("No se pudo calcular ROC-AUC en CV para este modelo/pipeline.")

# -------------------------
# Descargas
# -------------------------
st.subheader("⬇️ Descargas")

# 1) Dataset limpio (imputado simple) para referencia
df_download = df.copy()
num_cols_all, cat_cols_all = get_feature_types(df_download, target_cols=[])
df_download = impute_simple(df_download, num_cols_all, cat_cols_all)

csv_buffer = io.StringIO()
df_download.to_csv(csv_buffer, index=False)
st.download_button(
    "Descargar dataset 'limpio' (CSV, imputado simple)",
    data=csv_buffer.getvalue(),
    file_name="dataset_limpio.csv",
    mime="text/csv",
)

# 2) Predicciones del conjunto de prueba
preds_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
})
if y_proba is not None:
    preds_df["prob_1"] = y_proba

csv_buffer2 = io.StringIO()
preds_df.to_csv(csv_buffer2, index=False)
st.download_button(
    "Descargar predicciones (holdout)",
    data=csv_buffer2.getvalue(),
    file_name="predicciones_holdout.csv",
    mime="text/csv",
)

# -------------------------
# Ayuda
# -------------------------
with st.expander("❓ Cómo usar"):
    st.markdown(textwrap.dedent(f"""
    1. **Carga** un CSV/XLSX con una columna **`charges`**.
    2. Ajusta el **umbral** (por defecto se usa 10000 si es razonable).
    3. Elige las **features** (por defecto todas excepto `charges` y `contrata_seguro`).
    4. Selecciona el **modelo** e hiperparámetros mínimos.
    5. Define **holdout** y **K-Fold**.
    6. Revisa las métricas, la matriz de confusión y (si aplica) la **curva ROC**.
    7. Descarga el **dataset imputado** y las **predicciones** del holdout.
    """))

st.success("Listo. ¡Entrena, evalúa y descarga resultados!")
