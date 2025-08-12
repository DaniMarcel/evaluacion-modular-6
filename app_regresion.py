# app_regresion.py
# Streamlit app: Regresión para predecir 'charges' en insurance.csv
# Requisitos: streamlit, pandas, numpy, scikit-learn, matplotlib
# Ejecutar: streamlit run app_regresion.py

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
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


# -------------------------
# Configuración de página
# -------------------------
st.set_page_config(
    page_title="Regresión — insurance.csv",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Modelo de Regresión — Seguro Médico")
st.caption(
    "Predice **charges** con modelos simples y preprocesamiento automático (one-hot + escalado)."
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


def get_model_and_params(name: str, params: dict):
    if name == "LinearRegression":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(alpha=params.get("alpha", 1.0), random_state=None)
    if name == "Lasso":
        return Lasso(alpha=params.get("alpha", 0.001), max_iter=params.get("max_iter", 10000), random_state=None)
    if name == "KNeighborsRegressor":
        return KNeighborsRegressor(n_neighbors=params.get("n_neighbors", 5), weights=params.get("weights", "uniform"))
    if name == "DecisionTreeRegressor":
        return DecisionTreeRegressor(
            max_depth=params.get("max_depth", 5),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=params.get("random_state", 42),
        )
    if name == "Dummy (media)":
        return DummyRegressor(strategy="mean")
    return LinearRegression()


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_processed_feature_names(pre: ColumnTransformer) -> List[str]:
    """Intenta recuperar los nombres de features después del preprocesamiento."""
    out_names: List[str] = []
    for name, transformer, cols in pre.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps"):
            last = list(transformer.named_steps.values())[-1]
        else:
            last = transformer
        try:
            names = last.get_feature_names_out(cols)
        except Exception:
            if isinstance(cols, list):
                names = cols
            else:
                names = [cols]
        out_names.extend(list(names))
    return out_names


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

# Asegurar que 'charges' existe y es numérico
if "charges" not in df.columns:
    st.error("El dataset no contiene la columna **'charges'**, necesaria como objetivo de regresión.")
    st.stop()

df["charges"] = safe_coerce_numeric(df["charges"])

# Vista previa
with st.expander("📦 Vista previa de datos", expanded=False):
    st.write("**Shape:**", df.shape)
    st.dataframe(df.head(10), use_container_width=True)
    st.write("**Columnas:**", list(df.columns))

# -------------------------
# Selección de features y modelo
# -------------------------
drop_cols_default = ["charges"]
numeric_cols_all, categorical_cols_all = get_feature_types(df, target_cols=drop_cols_default)

all_features = [c for c in df.columns if c not in drop_cols_default]
feat_selected = st.multiselect(
    "Selecciona features (excluye 'charges')",
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
    ["LinearRegression", "Ridge", "Lasso", "KNeighborsRegressor", "DecisionTreeRegressor", "Dummy (media)"],
    index=0,
)

# Hiperparámetros
params = {}
if model_name in ("Ridge", "Lasso"):
    alpha = st.sidebar.number_input("alpha (regularización)", min_value=0.0001, max_value=1000.0, value=1.0, step=0.1)
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

# División holdout
st.sidebar.subheader("📐 Validación")
test_size = st.sidebar.slider("test_size (holdout)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
kfolds = st.sidebar.slider("K de KFold (CV)", min_value=3, max_value=10, value=5, step=1)
shuffle = st.sidebar.checkbox("shuffle en KFold", value=True)
random_state = st.sidebar.number_input("random_state", min_value=0, max_value=10_000, value=42, step=1)

# -------------------------
# Preparación de datos
# -------------------------
X = df[feat_selected].copy()
y = df["charges"].values

# Preprocesador
preprocessor = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols)

# Estimador y pipeline
estimator = get_model_and_params(model_name, {**params, "random_state": int(random_state)})
pipe = Pipeline(steps=[("pre", preprocessor), ("reg", estimator)])

# -------------------------
# Holdout
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse_val = rmse(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Resultados — Holdout")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:,.2f}")
col2.metric("RMSE", f"{rmse_val:,.2f}")
col3.metric("R²", f"{r2:.3f}")

# Visualizaciones de diagnóstico
# 1) y_true vs y_pred
fig1, ax1 = plt.subplots(figsize=(4, 3))
ax1.scatter(y_test, y_pred)
min_val = float(np.min([y_test.min(), y_pred.min()]))
max_val = float(np.max([y_test.max(), y_pred.max()]))
ax1.plot([min_val, max_val], [min_val, max_val], linestyle="--")
ax1.set_xlabel("Observado (y_test)")
ax1.set_ylabel("Predicho (y_pred)")
ax1.set_title("Observado vs. Predicho")
fig1.tight_layout()
st.pyplot(fig1, use_container_width=True)

# 2) Residuales
resid = y_test - y_pred
fig2, ax2 = plt.subplots(figsize=(4, 3))
ax2.hist(resid, bins=30)
ax2.set_title("Histograma de residuales")
ax2.set_xlabel("Residual (y - y_pred)")
fig2.tight_layout()
st.pyplot(fig2, use_container_width=True)

# 3) Importancias / Coeficientes (si disponible)
with st.expander("📌 Importancias / Coeficientes (top 20)"):
    try:
        # Recuperar nombres de features procesadas
        pre = pipe.named_steps["pre"]
        feat_names = get_processed_feature_names(pre)
        reg = pipe.named_steps["reg"]

        importances = None
        if hasattr(reg, "feature_importances_"):
            importances = np.asarray(reg.feature_importances_, dtype=float)
        elif hasattr(reg, "coef_"):
            coef = np.asarray(reg.coef_, dtype=float)
            importances = np.abs(coef).ravel()
        if importances is not None and len(importances) == len(feat_names):
            imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False).head(20)
            st.dataframe(imp_df, use_container_width=True)

            # Plot barras
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
            ax3.set_title("Top 20 importancias / |coef|")
            fig3.tight_layout()
            st.pyplot(fig3, use_container_width=True)
        else:
            st.info("El modelo no expone importancias/coeficientes compatibles o no se pudieron mapear.")
    except Exception as e:
        st.info(f"No se pudieron mostrar importancias/coeficientes: {e}")

# -------------------------
# Validación Cruzada (CV)
# -------------------------
st.subheader("🔁 Validación Cruzada (KFold)")
cv = KFold(n_splits=int(kfolds), shuffle=shuffle, random_state=int(random_state) if shuffle else None)

scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}

cv_results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=None, return_train_score=False)

mae_cv = -cv_results["test_MAE"].mean()
mae_cv_std = cv_results["test_MAE"].std()
rmse_cv = -cv_results["test_RMSE"].mean()
rmse_cv_std = cv_results["test_RMSE"].std()
r2_cv = cv_results["test_R2"].mean()
r2_cv_std = cv_results["test_R2"].std()

st.write(
    f"**MAE CV ({kfolds}-Fold)**: {mae_cv:,.2f} ± {mae_cv_std:,.2f}  \n"
    f"**RMSE CV ({kfolds}-Fold)**: {rmse_cv:,.2f} ± {rmse_cv_std:,.2f}  \n"
    f"**R² CV ({kfolds}-Fold)**: {r2_cv:.3f} ± {r2_cv_std:.3f}"
)

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
    "residual": resid,
})
csv_buffer2 = io.StringIO()
preds_df.to_csv(csv_buffer2, index=False)
st.download_button(
    "Descargar predicciones (holdout)",
    data=csv_buffer2.getvalue(),
    file_name="predicciones_holdout_regresion.csv",
    mime="text/csv",
)

# -------------------------
# Ayuda
# -------------------------
with st.expander("❓ Cómo usar"):
    st.markdown(textwrap.dedent(f"""
    1. **Carga** un CSV/XLSX con la columna **`charges`** (objetivo de regresión).
    2. Elige las **features** (por defecto todas salvo `charges`).
    3. Selecciona el **modelo** e hiperparámetros mínimos.
    4. Define **holdout** y **K-Fold**.
    5. Revisa métricas (MAE, RMSE, R²), y los gráficos de diagnóstico.
    6. Descarga el **dataset imputado** y las **predicciones** del holdout.
    """))

st.success("Listo. ¡Entrena, evalúa, visualiza y descarga resultados!")
