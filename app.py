import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostClassifier


# =========================
# Config
# =========================
st.set_page_config(page_title="IBOV Signal — Sistema Preditivo", layout="wide")

DEFAULT_CSV = "Dados Ibovespa (2).csv"
MODEL_PATH = "modelo_catboost.pkl"
SCALER_PATH = "scaler_minmax.pkl"
LOG_PATH = os.path.join("logs", "predictions_log.csv")


# =========================
# Funções do notebook
# =========================
def volume_to_float(value):
    if isinstance(value, str):
        value = value.strip().upper()
        factor = {"K": 1e3, "M": 1e6, "B": 1e9}
        suffix = value[-1]
        multiplier = factor.get(suffix, 1)
        value = value[:-1] if suffix in factor else value
        try:
            return float(value.replace(".", "").replace(",", ".")) * multiplier
        except:
            return np.nan
    return np.nan


def calculate_rsi(prices, window=14):
    changes = prices.diff()
    gains = changes.clip(lower=0)
    losses = -changes.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd_components(prices, short=12, long=26, signal=9):
    short_ema = prices.ewm(span=short, adjust=False).mean()
    long_ema = prices.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def obv_series(data):
    obv = [0]
    for i in range(1, len(data)):
        if data["Último"].iat[i] > data["Último"].iat[i - 1]:
            obv.append(obv[-1] + data["Vol."].iat[i])
        elif data["Último"].iat[i] < data["Último"].iat[i - 1]:
            obv.append(obv[-1] - data["Vol."].iat[i])
        else:
            obv.append(obv[-1])
    return obv


def zscore_roll(s: pd.Series, w: int = 20) -> pd.Series:
    m = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std()
    return (s - m) / sd


def corrige_escala_ultimo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df["Último"] = pd.to_numeric(df["Último"], errors="coerce")
    df = df.dropna(subset=["Data", "Último"]).sort_values("Data").reset_index(drop=True)

    for i in range(1, len(df)):
        prev = float(df.loc[i - 1, "Último"])
        curr = float(df.loc[i, "Último"])

        if curr < prev * 0.2:
            for fator in [10, 100, 1000]:
                if prev * 0.7 < curr * fator < prev * 1.3:
                    df.loc[i, "Último"] = curr * fator
                    break
    return df


def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv)
    df.columns = df.columns.str.strip()

    df["Data"] = pd.to_datetime(df["Data"], format="%d.%m.%Y", errors="coerce")
    if df["Data"].isna().mean() > 0.5:
        df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")

    df["Vol."] = df["Vol."].apply(volume_to_float)
    for coluna in ["Último", "Abertura", "Máxima", "Mínima"]:
        df[coluna] = (
            df[coluna].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df = corrige_escala_ultimo(df)
    df = df.sort_values("Data").dropna(subset=["Data", "Último"]).reset_index(drop=True)

    df["var_pct"] = df["Último"].pct_change()
    for dias in [3, 7, 14, 21, 30]:
        df[f"mm_{dias}"] = df["Último"].rolling(dias, min_periods=dias).mean()
    for dias in [5, 10, 20]:
        df[f"vol_{dias}"] = df["Último"].rolling(dias, min_periods=dias).std()

    df["desvio_mm3"] = df["Último"] - df["mm_3"]
    df["dia"] = df["Data"].dt.weekday
    df["rsi"] = calculate_rsi(df["Último"])

    macd, sinal, hist = macd_components(df["Último"])
    df["macd"], df["sinal_macd"], df["hist_macd"] = macd, sinal, hist

    bb_media = df["Último"].rolling(20, min_periods=20).mean()
    bb_std = df["Último"].rolling(20, min_periods=20).std()
    df["bb_media"] = bb_media
    df["bb_std"] = bb_std
    df["bb_sup"] = bb_media + 2 * bb_std
    df["bb_inf"] = bb_media - 2 * bb_std
    df["bb_largura"] = (df["bb_sup"] - df["bb_inf"]) / bb_media

    tr1 = df["Máxima"] - df["Mínima"]
    tr2 = (df["Máxima"] - df["Último"].shift(1)).abs()
    tr3 = (df["Mínima"] - df["Último"].shift(1)).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(14, min_periods=14).mean()

    df["obv"] = obv_series(df)
    df["Alvo"] = (df["Último"].shift(-1) > df["Último"]).astype("int8")
    df = df.iloc[:-1].copy()

    df["ret_1d"] = df["Último"].pct_change()
    df["log_ret"] = np.log(df["Último"]).diff()
    df["ret_5d"] = df["Último"].pct_change(5)
    df["rv_20"] = df["ret_1d"].rolling(20, min_periods=20).std()

    df["atr_pct"] = df["ATR"] / df["Último"]
    df["desvio_mm3_pct"] = (df["desvio_mm3"] / df["mm_3"]).replace([np.inf, -np.inf], np.nan)

    df["vol_log"] = np.log(df["Vol."].clip(lower=1))
    df["vol_ret"] = df["Vol."].pct_change().replace([np.inf, -np.inf], np.nan)

    df["obv_diff"] = pd.Series(df["obv"]).diff()

    df["z_close_20"] = zscore_roll(df["Último"], 20)
    df["z_rsi_20"] = zscore_roll(df["rsi"], 20)
    df["z_macd_20"] = zscore_roll(df["macd"], 20)

    features_sugeridas = [
        "ret_1d", "log_ret", "ret_5d", "rv_20",
        "atr_pct", "bb_largura", "desvio_mm3_pct",
        "vol_log", "vol_ret", "obv_diff",
        "rsi", "macd", "sinal_macd", "hist_macd",
        "dia", "z_close_20", "z_rsi_20", "z_macd_20"
    ]

    df = df.dropna(subset=features_sugeridas + ["Alvo"]).copy()
    df.attrs["features_sugeridas"] = features_sugeridas
    return df


def make_catboost():
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=10,
        grow_policy="Lossguide",
        border_count=64,
        eval_metric="F1",
        early_stopping_rounds=50,
        random_state=42,
        verbose=0,
    )


def timeseries_cv_f1(model_factory, X, y, n_splits=5):
    """
    CV temporal "na mão" (compatível com sklearn 1.6+ e CatBoost).
    Retorna lista de F1 por fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        Xtr, Xva = X[train_idx], X[val_idx]
        ytr, yva = y[train_idx], y[val_idx]
        m = model_factory()
        m.fit(Xtr, ytr)
        yp = m.predict(Xva)
        scores.append(f1_score(yva, yp))
    return scores


def ensure_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    header = ["timestamp", "source", "action", "selected_date", "pred_direction", "pred_proba", "threshold"]
    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        pd.DataFrame(columns=header).to_csv(LOG_PATH, index=False)
        return
    try:
        pd.read_csv(LOG_PATH, on_bad_lines="skip")
    except Exception:
        try:
            os.remove(LOG_PATH)
        except:
            pass
        pd.DataFrame(columns=header).to_csv(LOG_PATH, index=False)


def append_log(source, action, selected_date, pred_direction, pred_proba, threshold):
    ensure_log()
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "action": action,
        "selected_date": selected_date,
        "pred_direction": int(pred_direction) if pred_direction is not None else np.nan,
        "pred_proba": float(pred_proba) if pred_proba is not None else np.nan,
        "threshold": float(threshold),
    }])
    row.to_csv(LOG_PATH, mode="a", header=False, index=False)


# =========================
# UI
# =========================
st.title("📈 IBOV Signal — Sistema Preditivo (Interativo)")

with st.sidebar:
    st.header("Dados")
    uploaded = st.file_uploader("Upload de CSV (opcional)", type=["csv"])

    test_n = st.number_input("Janela de teste (últimos N) — use 30 p/ comparar com Colab", min_value=10, max_value=260, value=30, step=10)
    threshold = st.slider("Threshold P(ALTA) ≥ t (use 0.50 p/ comparar)", 0.30, 0.70, 0.50, 0.01)

    st.header("Re-treino")
    retrain = st.button("🔁 Re-treinar e salvar modelo + scaler (.pkl)")
    run_cv = st.checkbox("Rodar CV temporal (F1) no re-treino", value=True)

    st.header("Logs")
    show_logs = st.checkbox("Mostrar logs", value=False)


# =========================
# CSV
# =========================
if uploaded is not None:
    tmp_path = "tmp_upload.csv"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    csv_path = tmp_path
    source_name = f"UPLOAD:{uploaded.name}"
else:
    csv_path = DEFAULT_CSV
    source_name = f"REPO:{DEFAULT_CSV}"
    if not os.path.exists(csv_path):
        st.error(f"Não encontrei '{DEFAULT_CSV}' no repositório.")
        st.stop()

df = carregar_dados(csv_path)
features = df.attrs["features_sugeridas"]

X_raw = df[features].values
y_raw = df["Alvo"].values

split_idx = len(X_raw) - int(test_n)
if split_idx <= 0:
    st.error("Dataset pequeno demais para esse tamanho de teste.")
    st.stop()

X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
y_train, y_test = y_raw[:split_idx], y_raw[split_idx:]


# =========================
# Re-treino (para alinhar com Colab)
# =========================
if retrain:
    scaler = MinMaxScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)

    if run_cv:
        with st.sidebar:
            with st.spinner("Rodando CV temporal (F1)..."):
                f1_scores = timeseries_cv_f1(make_catboost, X_train, y_train, n_splits=5)
        st.sidebar.success(f"F1 médio (CV): {np.mean(f1_scores):.3f} (+/- {(np.std(f1_scores)*2):.3f})")

    model = make_catboost()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    st.sidebar.success("Salvei modelo_catboost.pkl e scaler_minmax.pkl (re-treinados).")

# =========================
# Load modelo/scaler (ou treina se faltar)
# =========================
try:
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else MinMaxScaler().fit(X_train_raw)
except:
    scaler = MinMaxScaler().fit(X_train_raw)

X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

try:
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else make_catboost()
    if not os.path.exists(MODEL_PATH):
        model.fit(X_train, y_train)
except:
    model = make_catboost()
    model.fit(X_train, y_train)


# =========================
# Avaliação
# =========================
y_pred_test = model.predict(X_test)
acc = accuracy_score(y_test, y_pred_test)
f1t = f1_score(y_test, y_pred_test)

st.subheader("✅ Avaliação (mesmo split temporal do Colab quando N=30)")
c1, c2, c3 = st.columns(3)
c1.metric("Acurácia (teste)", f"{acc:.2%}")
c2.metric("F1-score (teste)", f"{f1t:.3f}")
c3.metric("Tamanho teste", f"{len(y_test)}")

st.write("Confusão:")
st.write(confusion_matrix(y_test, y_pred_test))
st.text("Relatório:")
st.text(classification_report(y_test, y_pred_test))


# =========================
# Probabilidades/sinais no histórico
# =========================
X_all_scaled = scaler.transform(X_raw)
if hasattr(model, "predict_proba"):
    proba_all = model.predict_proba(X_all_scaled)[:, 1]
else:
    proba_all = model.predict(X_all_scaled).astype(float)

pred_all = (proba_all >= threshold).astype(int)


# =========================
# Produto (data -> previsão)
# =========================
st.divider()
st.subheader("🧠 Produto — selecione uma data e veja a tendência do dia seguinte")

date_options = df["Data"].dt.date.tolist()
selected_date = st.selectbox("Data (histórico)", options=date_options, index=len(date_options) - 1)

idx_list = df.index[df["Data"].dt.date == selected_date]
i = int(idx_list[0])

X_sel_raw = df.loc[[i], features].values
X_sel = scaler.transform(X_sel_raw)

if hasattr(model, "predict_proba"):
    p_sel = float(model.predict_proba(X_sel)[0, 1])
else:
    p_sel = float(model.predict(X_sel)[0])

y_sel = int(p_sel >= threshold)

if y_sel == 1:
    st.success(f"📈 Tendência prevista para o dia seguinte: **ALTA** (P(ALTA)={p_sel:.2%})")
else:
    st.warning(f"📉 Tendência prevista para o dia seguinte: **BAIXA** (P(ALTA)={p_sel:.2%})")

append_log(source_name, "predict_by_date", str(selected_date), y_sel, p_sel, threshold)

with st.expander("Ajustes do gráfico", expanded=True):
    view_n = st.slider("Mostrar últimos N pontos", 60, min(1500, len(df)), 400, 20)

df_plot = df.tail(int(view_n)).copy()
start_idx_plot = len(df) - len(df_plot)

proba_plot = proba_all[start_idx_plot:start_idx_plot + len(df_plot)]
pred_plot = pred_all[start_idx_plot:start_idx_plot + len(df_plot)]
price_vals = df_plot["Último"].astype(float).values

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    row_heights=[0.68, 0.32],
    subplot_titles=("Preço + Sinal", "Probabilidade de ALTA (com threshold)")
)

fig.add_trace(go.Scatter(x=df_plot["Data"], y=df_plot["Último"], mode="lines", name="Preço (Último)"), row=1, col=1)

fig.add_trace(
    go.Scatter(x=df_plot["Data"], y=np.where(pred_plot == 1, price_vals, np.nan), mode="markers",
               name="Sinal: ALTA", marker=dict(size=9, symbol="triangle-up")),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df_plot["Data"], y=np.where(pred_plot == 0, price_vals, np.nan), mode="markers",
               name="Sinal: BAIXA", marker=dict(size=8, symbol="triangle-down")),
    row=1, col=1
)

fig.add_trace(go.Scatter(x=df_plot["Data"], y=proba_plot, mode="lines", fill="tozeroy", name="P(ALTA)"), row=2, col=1)
fig.add_hline(y=threshold, line_dash="dash", line_width=2, annotation_text=f"threshold={threshold:.2f}", row=2, col=1)

fig.add_vline(x=pd.to_datetime(selected_date), line_width=2, row=1, col=1)
fig.add_vline(x=pd.to_datetime(selected_date), line_width=2, row=2, col=1)

fig.update_layout(height=650, margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h"))
fig.update_yaxes(title_text="Preço", row=1, col=1)
fig.update_yaxes(title_text="P(ALTA)", range=[0, 1], row=2, col=1)
fig.update_xaxes(rangeslider_visible=True)

st.plotly_chart(fig, use_container_width=True)


# =========================
# Logs
# =========================
if show_logs:
    st.divider()
    st.subheader("Logs do app")
    ensure_log()
    try:
        log_df = pd.read_csv(LOG_PATH, on_bad_lines="skip")
        st.dataframe(log_df.tail(100), use_container_width=True)
    except Exception as e:
        st.warning(f"Falha ao ler logs: {e}")
