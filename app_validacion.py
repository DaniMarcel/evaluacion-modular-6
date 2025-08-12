# app_validacion.py
# Streamlit app: Comparación de técnicas de Validación Cruzada en insurance.csv
# Requisitos: streamlit, pandas, numpy, scikit-learn, matplotlib
# Ejecutar: streamlit run app_validacion.py

from __future__ import annotations

import io
import textwrap
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
    LeaveOneOut,
    cross_validate,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


# -------------------------
# Configuración de página
# -------------------------
st.set_page_config(
    page_title="Validación Cruzada — insurance.csv",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔁 Validación Cruzada — Ejemplos Prácticos")
st.caption("Compara Holdout, K-Fold, Shuffle Split y Leave-One-Out (LOOCV) en modelos simples.")


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


def compute_metrics_regression(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def compute_metrics_classification(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            out["ROC-AUC"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["ROC-AUC"] = float("nan")
    else:
        out["ROC-AUC"] = float("nan")
    return out


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
            xls = pd.ExcelFile(uploaded)
            sheet_name = st.sidebar.selectbox("Hoja de Excel", xls.sheet_names, index=0)
            df = load_excel(uploaded, sheet_name=sheet_name)
        else:
            df = load_csv(uploaded, sep=sep, encoding=encoding)
    else:
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
# Tipo de problema
# -------------------------
problem_type = st.sidebar.radio("Tipo de problema", ["Regresión (charges)", "Clasificación (charges > umbral)"], index=0)

# Validaciones y preparación del target
if "charges" not in df.columns:
    st.error("El dataset no contiene la columna **'charges'** requerida.")
    st.stop()

df["charges"] = safe_coerce_numeric(df["charges"])

if problem_type.startswith("Clasificación"):
    min_c = float(np.nanpercentile(df["charges"], 1))
    max_c = float(np.nanpercentile(df["charges"], 99))
    default_thr = 10000.0 if min_c <= 10000 <= max_c else float(np.nanmedian(df["charges"]))
    threshold = st.sidebar.number_input(
        "Umbral para binarizar: 1 si charges > umbral",
        min_value=float(np.floor(min_c)) if np.isfinite(min_c) else 0.0,
        max_value=float(np.ceil(max_c)) if np.isfinite(max_c) else 100000.0,
        value=float(default_thr),
        step=100.0,
    )
    df["target_bin"] = (df["charges"] > threshold).astype(int)
    y = df["target_bin"].astype(int).values
else:
    y = df["charges"].values

# -------------------------
# Vista previa
# -------------------------
with st.expander("📦 Vista previa de datos", expanded=False):
    st.write("**Shape:**", df.shape)
    st.dataframe(df.head(10), use_container_width=True)
    st.write("**Columnas:**", list(df.columns))

# -------------------------
# Selección de features y modelo
# -------------------------
drop_cols_default = ["charges", "target_bin"] if "target_bin" in df.columns else ["charges"]
numeric_cols_all, categorical_cols_all = get_feature_types(df, target_cols=drop_cols_default)

all_features = [c for c in df.columns if c not in drop_cols_default]
feat_selected = st.multiselect(
    "Selecciona features (excluye el objetivo)",
    options=all_features,
    default=all_features,
)

if not feat_selected:
    st.error("Selecciona al menos una feature para entrenar el modelo.")
    st.stop()

# Tipos de columnas en selección
num_cols = [c for c in numeric_cols_all if c in feat_selected]
cat_cols = [c for c in categorical_cols_all if c in feat_selected]

# Modelos según problema
st.sidebar.subheader("🤖 Modelo")
if problem_type.startswith("Regresión"):
    model_name = st.sidebar.selectbox(
        "Algoritmo",
        ["LinearRegression", "Ridge", "Lasso", "KNeighborsRegressor", "DecisionTreeRegressor", "Dummy (media)"],
        index=0,
    )
else:
    model_name = st.sidebar.selectbox(
        "Algoritmo",
        ["LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier", "Dummy (most_frequent)"],
        index=0,
    )

# Hiperparámetros mínimos
params = {}
if model_name in ("Ridge", "Lasso"):
    alpha = st.sidebar.number_input("alpha", min_value=0.0001, max_value=1000.0, value=1.0, step=0.1)
    params.update(dict(alpha=float(alpha)))
    if model_name == "Lasso":
        max_iter = st.sidebar.number_input("max_iter", min_value=1000, max_value=50000, value=10000, step=1000)
        params.update(dict(max_iter=int(max_iter)))
elif model_name == "KNeighborsRegressor":
    n_neighbors = st.sidebar.slider("n_neighbors", min_value=1, max_value=50, value=5, step=1)
    weights = st.sidebar.selectbox("weights", ["uniform", "distance"], index=0)
    params.update(dict(n_neighbors=int(n_neighbors), weights=weights))
elif model_name == "DecisionTreeRegressor":
    max_depth = st.sidebar.slider("max_depth", min_value=1, max_value=50, value=5, step=1)
    min_samples_split = st.sidebar.slider("min_samples_split", min_value=2, max_value=20, value=2, step=1)
    params.update(dict(max_depth=int(max_depth), min_samples_split=int(min_samples_split)))
elif model_name == "LogisticRegression":
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
elif model_name.startswith("Dummy"):
    dummy_strategy = "mean" if problem_type.startswith("Regresión") else "most_frequent"
    st.sidebar.info(f"Estrategia dummy: {dummy_strategy}")

# Validación — parámetros globales
st.sidebar.subheader("📐 Validación")
test_size = st.sidebar.slider("test_size (Holdout)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
kfolds = st.sidebar.slider("K (K-Fold)", min_value=3, max_value=10, value=5, step=1)
n_splits_shuffle = st.sidebar.slider("n_splits (ShuffleSplit)", min_value=3, max_value=20, value=5, step=1)
test_size_shuffle = st.sidebar.slider("test_size (ShuffleSplit)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
random_state = st.sidebar.number_input("random_state", min_value=0, max_value=10_000, value=42, step=1)

# -------------------------
# Preparación de datos y modelo
# -------------------------
X = df[feat_selected].copy()

# Preprocesador (común)
preprocessor = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols)

# Instanciar modelo
if problem_type.startswith("Regresión"):
    if model_name == "LinearRegression":
        estimator = LinearRegression()
    elif model_name == "Ridge":
        estimator = Ridge(**params)
    elif model_name == "Lasso":
        estimator = Lasso(**params)
    elif model_name == "KNeighborsRegressor":
        estimator = KNeighborsRegressor(**params)
    elif model_name == "DecisionTreeRegressor":
        estimator = DecisionTreeRegressor(**params, random_state=int(random_state))
    else:
        estimator = DummyRegressor(strategy="mean")
else:
    if model_name == "LogisticRegression":
        estimator = LogisticRegression(**params)
    elif model_name == "KNeighborsClassifier":
        estimator = KNeighborsClassifier(**params)
    elif model_name == "DecisionTreeClassifier":
        estimator = DecisionTreeClassifier(**params, random_state=int(random_state))
    else:
        estimator = DummyClassifier(strategy="most_frequent", random_state=int(random_state))

pipe = Pipeline(steps=[("pre", preprocessor), ("est", estimator)])

# -------------------------
# A) Holdout
# -------------------------
st.subheader("A) Holdout")
if problem_type.startswith("Clasificación"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state), stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

if problem_type.startswith("Regresión"):
    metrics_holdout = compute_metrics_regression(y_test, y_pred)
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics_holdout['MAE']:,.2f}")
    col2.metric("RMSE", f"{metrics_holdout['RMSE']:,.2f}")
    col3.metric("R²", f"{metrics_holdout['R2']:.3f}")

    # Plot observado vs predicho
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.scatter(y_test, y_pred)
    min_val = float(np.min([y_test.min(), y_pred.min()]))
    max_val = float(np.max([y_test.max(), y_pred.max()]))
    ax1.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax1.set_xlabel("Observado (y_test)")
    ax1.set_ylabel("Predicho (y_pred)")
    ax1.set_title("Holdout — Observado vs. Predicho")
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=True)

    # Hist residuales
    resid = y_test - y_pred
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.hist(resid, bins=30)
    ax2.set_title("Holdout — Histograma de residuales")
    ax2.set_xlabel("y - y_pred")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
else:
    y_proba = probas_from_estimator(pipe, X_test)
    metrics_holdout = compute_metrics_classification(y_test, y_pred, y_proba)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{metrics_holdout['Accuracy']:.3f}")
    col2.metric("Precision", f"{metrics_holdout['Precision']:.3f}")
    col3.metric("Recall", f"{metrics_holdout['Recall']:.3f}")
    col4.metric("F1", f"{metrics_holdout['F1']:.3f}")
    col5.metric("ROC-AUC", f"{metrics_holdout['ROC-AUC']:.3f}" if np.isfinite(metrics_holdout['ROC-AUC']) else "—")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    im = ax_cm.imshow(cm, interpolation="nearest")
    ax_cm.figure.colorbar(im, ax=ax_cm)
    tick_marks = np.arange(2)
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels(["0", "1"])
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels(["0", "1"])
    ax_cm.set_ylabel("Real")
    ax_cm.set_xlabel("Predicho")
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig_cm.tight_layout()
    st.pyplot(fig_cm, use_container_width=True)

    # Curva ROC
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
        ax_roc.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title("Holdout — Curva ROC")
        ax_roc.legend(loc="lower right")
        fig_roc.tight_layout()
        st.pyplot(fig_roc, use_container_width=True)

    with st.expander("📃 Classification Report (holdout)"):
        report = classification_report(y_test, y_pred, digits=3, zero_division=0)
        st.code(report, language="text")


# -------------------------
# B) K-Fold
# -------------------------
st.subheader("B) K-Fold")
if problem_type.startswith("Regresión"):
    cv_kfold = KFold(n_splits=int(kfolds), shuffle=True, random_state=int(random_state))
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }
else:
    cv_kfold = StratifiedKFold(n_splits=int(kfolds), shuffle=True, random_state=int(random_state))
    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC-AUC": "roc_auc",
    }

try:
    cv_res_kfold = cross_validate(pipe, X, y, cv=cv_kfold, scoring=scoring, n_jobs=None, return_train_score=False)
    # Convertir a positivos los que sean negativos
    results_kfold = {}
    for key, vals in cv_res_kfold.items():
        if not key.startswith("test_"):
            continue
        name = key.replace("test_", "")
        arr = np.array(vals, dtype=float)
        if problem_type.startswith("Regresión") and name in ("MAE", "RMSE"):
            arr = -arr
        results_kfold[name] = (float(arr.mean()), float(arr.std()))
except Exception as e:
    st.error(f"No se pudo ejecutar K-Fold: {e}")
    results_kfold = {}

# -------------------------
# C) Shuffle Split
# -------------------------
st.subheader("C) Shuffle Split")
if problem_type.startswith("Regresión"):
    cv_shuffle = ShuffleSplit(n_splits=int(n_splits_shuffle), test_size=float(test_size_shuffle), random_state=int(random_state))
    scoring_shuffle = scoring  # mismo diccionario
else:
    cv_shuffle = StratifiedShuffleSplit(n_splits=int(n_splits_shuffle), test_size=float(test_size_shuffle), random_state=int(random_state))
    scoring_shuffle = scoring  # mismo diccionario

try:
    cv_res_shuffle = cross_validate(pipe, X, y, cv=cv_shuffle, scoring=scoring_shuffle, n_jobs=None, return_train_score=False)
    results_shuffle = {}
    for key, vals in cv_res_shuffle.items():
        if not key.startswith("test_"):
            continue
        name = key.replace("test_", "")
        arr = np.array(vals, dtype=float)
        if problem_type.startswith("Regresión") and name in ("MAE", "RMSE"):
            arr = -arr
        results_shuffle[name] = (float(arr.mean()), float(arr.std()))
except Exception as e:
    st.error(f"No se pudo ejecutar ShuffleSplit: {e}")
    results_shuffle = {}

# -------------------------
# D) LOOCV
# -------------------------
st.subheader("D) Leave-One-Out (LOOCV)")
n_samples = len(X)
if n_samples > 300:
    st.info("LOOCV es costoso. Limitaré a 300 filas para ejemplo (muestra aleatoria reproducible).")
    sample_idx = np.random.RandomState(int(random_state)).choice(np.arange(n_samples), size=300, replace=False)
    X_loocv = X.iloc[sample_idx].copy()
    y_loocv = y[sample_idx]
else:
    X_loocv = X.copy()
    y_loocv = y

if problem_type.startswith("Regresión"):
    cv_loo = LeaveOneOut()
    scoring_loo = scoring  # mismo diccionario
else:
    cv_loo = LeaveOneOut()
    scoring_loo = scoring  # intentará accuracy, precision, recall, f1 y roc_auc (puede fallar roc_auc)

try:
    cv_res_loo = cross_validate(pipe, X_loocv, y_loocv, cv=cv_loo, scoring=scoring_loo, n_jobs=None, return_train_score=False)
    results_loo = {}
    for key, vals in cv_res_loo.items():
        if not key.startswith("test_"):
            continue
        name = key.replace("test_", "")
        arr = np.array(vals, dtype=float)
        if problem_type.startswith("Regresión") and name in ("MAE", "RMSE"):
            arr = -arr
        results_loo[name] = (float(arr.mean()), float(arr.std()))
except Exception as e:
    st.info(f"No se pudo ejecutar LOOCV para todas las métricas: {e}")
    results_loo = {}

# -------------------------
# Tabla comparativa
# -------------------------
st.subheader("📋 Comparación de resultados")
if problem_type.startswith("Regresión"):
    metric_list = ["MAE", "RMSE", "R2"]
else:
    metric_list = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

def to_row(method_name: str, res: Dict[str, Tuple[float, float]]) -> Dict[str, str]:
    row = {"Método": method_name}
    for m in metric_list:
        if m in res:
            mean, std = res[m]
            row[m] = f"{mean:.3f} ± {std:.3f}"
        else:
            row[m] = "—"
    return row

rows = []
# Holdout (std vacío)
if problem_type.startswith("Regresión"):
    rows.append({"Método": "Holdout", "MAE": f"{metrics_holdout['MAE']:.3f}", "RMSE": f"{metrics_holdout['RMSE']:.3f}", "R2": f"{metrics_holdout['R2']:.3f}"})
else:
    rows.append({
        "Método": "Holdout",
        "Accuracy": f"{metrics_holdout['Accuracy']:.3f}",
        "Precision": f"{metrics_holdout['Precision']:.3f}",
        "Recall": f"{metrics_holdout['Recall']:.3f}",
        "F1": f"{metrics_holdout['F1']:.3f}",
        "ROC-AUC": f"{metrics_holdout['ROC-AUC']:.3f}" if np.isfinite(metrics_holdout['ROC-AUC']) else "—"
    })

rows.append(to_row("K-Fold", results_kfold))
rows.append(to_row("ShuffleSplit", results_shuffle))
rows.append(to_row("LOOCV", results_loo))

table_df = pd.DataFrame(rows)
st.dataframe(table_df, use_container_width=True)

# Boxplots por métrica (CV)
with st.expander("📦 Distribuciones CV (boxplots por método)"):
    if problem_type.startswith("Regresión"):
        # Construir arrays por método desde cross_validate
        def gather(arr_dict, name):
            vals = []
            if name == "MAE" and "test_MAE" in cv_res_kfold:
                vals.append(-np.array(cv_res_kfold["test_MAE"], dtype=float))
            if name == "RMSE" and "test_RMSE" in cv_res_kfold:
                vals.append(-np.array(cv_res_kfold["test_RMSE"], dtype=float))
            if name == "R2" and "test_R2" in cv_res_kfold:
                vals.append(np.array(cv_res_kfold["test_R2"], dtype=float))
            if name == "MAE" and "test_MAE" in cv_res_shuffle:
                vals.append(-np.array(cv_res_shuffle["test_MAE"], dtype=float))
            if name == "RMSE" and "test_RMSE" in cv_res_shuffle:
                vals.append(-np.array(cv_res_shuffle["test_RMSE"], dtype=float))
            if name == "R2" and "test_R2" in cv_res_shuffle:
                vals.append(np.array(cv_res_shuffle["test_R2"], dtype=float))
            return vals

        for metric in ["MAE", "RMSE", "R2"]:
            vals = gather(locals(), metric)
            if len(vals) == 0:
                continue
            labels = ["K-Fold", "ShuffleSplit"][:len(vals)]
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.boxplot(vals, labels=labels, showmeans=True)
            ax.set_title(f"Distribución {metric} (CV)")
            st.pyplot(fig, use_container_width=True)
    else:
        def gather_cls(name):
            vals = []
            if name in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC"):
                if f"test_{name}" in cv_res_kfold:
                    vals.append(np.array(cv_res_kfold[f"test_{name}"], dtype=float))
                if f"test_{name}" in cv_res_shuffle:
                    vals.append(np.array(cv_res_shuffle[f"test_{name}"], dtype=float))
            return vals

        for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            vals = gather_cls(metric)
            if len(vals) == 0:
                continue
            labels = ["K-Fold", "ShuffleSplit"][:len(vals)]
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.boxplot(vals, labels=labels, showmeans=True)
            ax.set_title(f"Distribución {metric} (CV)")
            st.pyplot(fig, use_container_width=True)

# -------------------------
# Descargas
# -------------------------
st.subheader("⬇️ Descargas")
# 1) Tabla comparativa
csv_buf = io.StringIO()
table_df.to_csv(csv_buf, index=False)
st.download_button(
    "Descargar tabla comparativa (CSV)",
    data=csv_buf.getvalue(),
    file_name="comparacion_validacion.csv",
    mime="text/csv",
)

# 2) Reporte rápido en Markdown
md = io.StringIO()
md.write("# Reporte de Validación Cruzada\n\n")
md.write(f"- **Problema**: {problem_type}\n")
md.write(f"- **Modelo**: {model_name}\n")
md.write(f"- **Features**: {', '.join(feat_selected)}\n")
if problem_type.startswith("Clasificación"):
    md.write(f"- **Umbral**: charges > {threshold}\n")
md.write("\n## Resultados Holdout\n\n")
if problem_type.startswith("Regresión"):
    md.write(f"- MAE: {metrics_holdout['MAE']:.3f}\n")
    md.write(f"- RMSE: {metrics_holdout['RMSE']:.3f}\n")
    md.write(f"- R²: {metrics_holdout['R2']:.3f}\n")
else:
    md.write(f"- Accuracy: {metrics_holdout['Accuracy']:.3f}\n")
    md.write(f"- Precision: {metrics_holdout['Precision']:.3f}\n")
    md.write(f"- Recall: {metrics_holdout['Recall']:.3f}\n")
    md.write(f"- F1: {metrics_holdout['F1']:.3f}\n")
    auc_txt = f"{metrics_holdout['ROC-AUC']:.3f}" if np.isfinite(metrics_holdout['ROC-AUC']) else "—"
    md.write(f"- ROC-AUC: {auc_txt}\n")

md.write("\n## Resumen CV\n\n")
md.write(table_df.to_markdown(index=False))
st.download_button(
    "Descargar reporte (Markdown)",
    data=md.getvalue(),
    file_name="reporte_validacion.md",
    mime="text/markdown",
)

# -------------------------
# Ayuda
# -------------------------
with st.expander("❓ Cómo usar"):
    st.markdown(textwrap.dedent("""
    1. **Carga** un CSV/XLSX con columna `charges` (para regresión) o úsala para crear una etiqueta binaria (clasificación).
    2. Elige **features**, **modelo** e **hiperparámetros** mínimos.
    3. Compara **Holdout**, **K-Fold**, **ShuffleSplit** y **LOOCV**.
    4. Revisa métricas y distribuciones de CV; descarga la tabla y un reporte rápido.
    """))

st.success("Listo. ¡Compara estrategias de validación y elige la mejor para tu caso!")
