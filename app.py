import os
import json
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import joblib


# =========================
# Config
# =========================
st.set_page_config(page_title="IBOV Signal — Sistema Preditivo", layout="wide")

DEFAULT_CSV = "Dados Ibovespa (2).csv"
MODEL_PATH = "modelo_catboost.pkl"
SCALER_PATH = "scaler_minmax.pkl"

LOG_DIR = "logs"
LOG_CSV_PATH = os.path.join(LOG_DIR, "usage_log.csv")
LOG_JSONL_PATH = os.path.join(LOG_DIR, "usage_log.jsonl")


# =========================
# MÉTRICAS FIXAS (do Colab) — SEM RETREINO
# =========================
METRICAS_COLAB = {
    "modelo": "CatBoostClassifier (treinado no Colab / Fase 2)",
    "janela_validacao": "Holdout temporal: últimos 30 registros como teste",
    "cv_f1_mean": 0.531,
    "cv_f1_pm": 0.083,
    "acc_train": 0.8203,
    "acc_test": 0.8000,
    "overfit": 0.0203,
    "cm": [[13, 3], [3, 11]],
    "report": """precision    recall  f1-score   support

0       0.81      0.81      0.81        16
1       0.79      0.79      0.79        14

accuracy                           0.80        30
macro avg       0.80      0.80      0.80        30
weighted avg    0.80      0.80      0.80        30"""
}


# =========================
# LOG DE USO (CSV + JSONL)
# =========================
def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"sess_{np.random.randint(10**8, 10**9)}"
    return st.session_state.session_id


def append_usage_log(event: dict):
    try:
        _ensure_log_dir()
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_id": _get_session_id(),
            **event,
        }

        with open(LOG_JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        df_row = pd.DataFrame([payload])
        if os.path.exists(LOG_CSV_PATH):
            df_row.to_csv(LOG_CSV_PATH, mode="a", header=False, index=False, encoding="utf-8")
        else:
            df_row.to_csv(LOG_CSV_PATH, mode="w", header=True, index=False, encoding="utf-8")
    except Exception:
        pass


# =========================
# Funções de features
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
        except Exception:
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
    vol = pd.to_numeric(data["Vol."], errors="coerce").fillna(0.0).values
    close = pd.to_numeric(data["Último"], errors="coerce").values
    obv = [0.0]
    for i in range(1, len(data)):
        if close[i] > close[i - 1]:
            obv.append(obv[-1] + vol[i])
        elif close[i] < close[i - 1]:
            obv.append(obv[-1] - vol[i])
        else:
            obv.append(obv[-1])
    return obv


def zscore_roll(s: pd.Series, w: int = 20, eps: float = 1e-6) -> pd.Series:
    """
    Em cenários "constantes", o desvio padrão pode virar 0 -> zscore ficava NaN.
    Aqui, quando std ~ 0, usamos eps, então o zscore fica ~0.
    """
    m = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std()
    sd = sd.mask(sd < eps, eps)
    return (s - m) / sd


def correcao_escala_por_vizinhanca(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df["Último"] = pd.to_numeric(df["Último"], errors="coerce")
    df = df.dropna(subset=["Data", "Último"]).sort_values("Data").reset_index(drop=True)

    for i in range(1, len(df)):
        prev = df.loc[i - 1, "Último"]
        curr = df.loc[i, "Último"]
        if pd.notna(prev) and pd.notna(curr) and curr < prev * 0.2:
            for fator in [10, 100, 1000]:
                if prev * 0.7 < curr * fator < prev * 1.3:
                    df.loc[i, "Último"] = curr * fator
                    break
    return df


def compute_features_inplace(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df = df.dropna(subset=["Data"]).sort_values("Data").reset_index(drop=True)

    for c in ["Último", "Abertura", "Máxima", "Mínima", "Vol."]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

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

    return df


def carregar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv)
    df.columns = df.columns.str.strip()

    df["Data"] = pd.to_datetime(df["Data"], format="%d.%m.%Y", errors="coerce")
    df = df.dropna(subset=["Data"]).sort_values("Data")

    df["Vol."] = df["Vol."].apply(volume_to_float)

    for coluna in ["Último", "Abertura", "Máxima", "Mínima"]:
        df[coluna] = (
            df[coluna].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df = correcao_escala_por_vizinhanca(df).sort_values("Data").reset_index(drop=True)
    df = compute_features_inplace(df)

    df["Alvo"] = (df["Último"].shift(-1) > df["Último"]).astype("int8")
    df = df.iloc[:-1].copy()

    features_sugeridas = [
        "ret_1d", "log_ret", "ret_5d", "rv_20",
        "atr_pct", "bb_largura", "desvio_mm3_pct",
        "vol_log", "vol_ret", "obv_diff",
        "rsi", "macd", "sinal_macd", "hist_macd",
        "dia", "z_close_20", "z_rsi_20", "z_macd_20"
    ]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features_sugeridas + ["Alvo"]).copy()
    df.attrs["features_sugeridas"] = features_sugeridas
    return df


# =========================
# ✅ NOVO: carregar CSV enviado (mesma lógica, sem cache e sem mexer nas abas atuais)
# =========================
def carregar_dados_upload(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Data: tenta formatos comuns
    if "Data" not in df.columns:
        raise ValueError("O CSV enviado precisa ter a coluna 'Data'.")

    df["Data"] = pd.to_datetime(df["Data"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Data"]).sort_values("Data")

    # Vol.: pode vir numérico ou como '10.2M', '350K', etc.
    if "Vol." not in df.columns:
        raise ValueError("O CSV enviado precisa ter a coluna 'Vol.'.")

    df["Vol."] = df["Vol."].apply(lambda v: v if pd.api.types.is_number(v) else volume_to_float(v))

    # Preços
    for coluna in ["Último", "Abertura", "Máxima", "Mínima"]:
        if coluna not in df.columns:
            raise ValueError(f"O CSV enviado precisa ter a coluna '{coluna}'.")
        df[coluna] = (
            df[coluna].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df = correcao_escala_por_vizinhanca(df).sort_values("Data").reset_index(drop=True)
    df = compute_features_inplace(df)

    df["Alvo"] = (df["Último"].shift(-1) > df["Último"]).astype("int8")
    df = df.iloc[:-1].copy()

    features_sugeridas = [
        "ret_1d", "log_ret", "ret_5d", "rv_20",
        "atr_pct", "bb_largura", "desvio_mm3_pct",
        "vol_log", "vol_ret", "obv_diff",
        "rsi", "macd", "sinal_macd", "hist_macd",
        "dia", "z_close_20", "z_rsi_20", "z_macd_20"
    ]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features_sugeridas + ["Alvo"]).copy()
    df.attrs["features_sugeridas"] = features_sugeridas
    return df


# =========================
# Cache
# =========================
@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Não encontrei `{MODEL_PATH}` no repositório.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Não encontrei `{SCALER_PATH}` no repositório.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_df_and_features(csv_path):
    df = carregar_dados(csv_path)
    features = df.attrs["features_sugeridas"]
    return df, features


# =========================
# Predição
# =========================
def predict_proba_batch(model, scaler, X, threshold):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Xs = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)[:, 1]
    else:
        proba = model.predict(Xs).astype(float)

    pred = (proba >= threshold).astype(int)
    return pred, proba


# =========================
# Gráfico Intuitivo (1 painel, 2 eixos)
# =========================
def make_signal_chart_intuitivo(
    df_plot: pd.DataFrame,
    pred,
    proba,
    threshold: float,
    title: str,
    height: int = 560,
    show_rangeslider: bool = True,
):
    df_plot = df_plot.copy()
    df_plot["Data"] = pd.to_datetime(df_plot["Data"], errors="coerce")
    df_plot = df_plot.dropna(subset=["Data"]).sort_values("Data")

    pred = np.asarray(pred, dtype=int)
    proba = np.asarray(proba, dtype=float)

    n = min(len(df_plot), len(pred), len(proba))
    df_plot = df_plot.iloc[:n].copy()
    pred = pred[:n]
    proba = proba[:n]

    dates = df_plot["Data"]
    price = df_plot["Último"].astype(float).values

    y_alta = np.where(pred == 1, price, np.nan)
    y_baixa = np.where(pred == 0, price, np.nan)

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=dates, y=price,
            mode="lines",
            name="Preço (Último)",
            line=dict(width=2),
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Preço: %{y:,.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=dates, y=y_alta,
            mode="markers",
            name="Sinal: ALTA",
            marker=dict(size=9, symbol="triangle-up"),
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br><b>Sinal:</b> ALTA<br>Preço: %{y:,.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=dates, y=y_baixa,
            mode="markers",
            name="Sinal: BAIXA",
            marker=dict(size=8, symbol="triangle-down"),
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br><b>Sinal:</b> BAIXA<br>Preço: %{y:,.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=dates, y=proba,
            mode="lines",
            name="P(ALTA)",
            yaxis="y2",
            line=dict(width=2),
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>P(ALTA): %{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[dates.min(), dates.max()],
            y=[threshold, threshold],
            mode="lines",
            name=f"threshold={threshold:.2f}",
            yaxis="y2",
            line=dict(width=2, dash="dash"),
            hoverinfo="skip",
        )
    )

    # ✅ legenda embaixo e visível
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=int(height),
        margin=dict(l=10, r=10, t=70, b=150),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=-0.28,
            yanchor="top",
            yref="paper",
            itemwidth=90,
        ),
        xaxis=dict(
            type="date",
            rangeslider=dict(visible=bool(show_rangeslider)),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=12, label="12m", step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            ),
        ),
        yaxis=dict(title="Preço", showgrid=True),
        yaxis2=dict(
            title="P(ALTA)",
            overlaying="y",
            side="right",
            range=[0, 1],
            showgrid=False,
        ),
    )

    fig.update_xaxes(range=[dates.min(), dates.max()])
    return fig


# =========================
# App
# =========================
st.title("📈 IBOV Signal — Sistema Preditivo (modelo do Colab, sem re-treino)")

with st.expander("ℹ️ Como usar o aplicativo (rápido)", expanded=True):
    st.markdown(
        """
- Este app usa um **modelo já treinado** para estimar a **probabilidade do IBOV subir no próximo dia** (**P(ALTA)**).
- Você ajusta o **Threshold** (na lateral). Se **P(ALTA) ≥ Threshold**, o sinal vira **ALTA**; caso contrário, **BAIXA**.

**Abas**
- **🧠 Produto (Simulação futura):** escolha uma **data futura** e um **cenário de simulação**. O app **simula preços até a data** e calcula o sinal/probabilidade para esse período (**não é dado real futuro**, é simulação).
- **📅 Histórico:** selecione uma **data do dataset** e veja a previsão para o **dia seguinte**, com gráfico do histórico.
- **🔎 Diagnóstico:** painel com **métricas do modelo** (fixas do treino) e informações do dataset.
- **📤 Entrada de Dados:** envie seu **CSV** (histórico) **ou** crie uma **linha manual** (OHLCV) e veja a previsão.
        """.strip()
    )

with st.sidebar:
    st.header("Config do Modelo")
    threshold = st.slider("Threshold para ALTA", 0.30, 0.70, 0.50, 0.01)

    st.divider()
    st.header("Config do Gráfico")
    view_n = st.slider("Janela do histórico — últimos N", 60, 1500, 400, 20)
    chart_height = st.slider("Altura do gráfico", 420, 900, 560, 10)
    show_rangeslider = st.checkbox("Mostrar range slider", value=True)

    st.divider()
    with st.expander("📌 Entenda P(ALTA) e Threshold", expanded=False):
        st.markdown(
            """
- **P(ALTA)**: probabilidade estimada de o índice fechar **mais alto** no **próximo dia**.
- **Threshold**: “linha de corte” para o sinal:
  - **P(ALTA) ≥ Threshold** → **ALTA**
  - **P(ALTA) < Threshold** → **BAIXA**
            """.strip()
        )

    st.divider()
    st.subheader("Log de uso")
    st.caption("Arquivos: logs/usage_log.csv e logs/usage_log.jsonl")

    if os.path.exists(LOG_CSV_PATH):
        with open(LOG_CSV_PATH, "rb") as f:
            st.download_button("⬇️ Baixar log CSV", data=f, file_name="usage_log.csv", mime="text/csv")
    if os.path.exists(LOG_JSONL_PATH):
        with open(LOG_JSONL_PATH, "rb") as f:
            st.download_button("⬇️ Baixar log JSONL", data=f, file_name="usage_log.jsonl", mime="application/jsonl")

if not os.path.exists(DEFAULT_CSV):
    st.error(
        f"Não encontrei `{DEFAULT_CSV}` no repositório. "
        "Suba esse CSV junto com o app.py para o Streamlit não pedir upload."
    )
    st.stop()

try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(str(e))
    st.stop()

df, features = load_df_and_features(DEFAULT_CSV)

# ✅ sem alterar as abas atuais, apenas adicionando uma nova aba no final
tab_produto, tab_historico, tab_diag, tab_entrada = st.tabs(
    ["🧠 Produto (Simulação futura)", "📅 Histórico (data do dataset)", "🔎 Diagnóstico (métricas)", "📤 Entrada de Dados"]
)

# =========================
# TAB 1 — PRODUTO (SIMULAÇÃO FUTURA)
# =========================
with tab_produto:
    st.subheader("Produto: Simulação futura (data manual, sem travar)")
    st.info(
        "Aqui você escolhe uma **data futura** e um **cenário**. Como não existe preço real do futuro no CSV, "
        "o app **simula uma trajetória de preços** até a data escolhida e calcula **P(ALTA)** e **Sinal** "
        f"(com base no **Threshold** definido na lateral).",
        icon="ℹ️",
    )

    last_date = pd.to_datetime(df["Data"].iloc[-1])
    last_price = float(df["Último"].iloc[-1])
    last_vol = float(df["Vol."].iloc[-1]) if pd.notna(df["Vol."].iloc[-1]) else 0.0

    st.info(
        f"Último ponto: {last_date.date()} — Último={last_price:,.2f}"
        .replace(",", "X").replace(".", ",").replace("X", ".")
    )

    alvo = st.date_input(
        "Digite/Selecione a data futura",
        value=(last_date + timedelta(days=30)).date(),
        key="alvo_date",
    )
    if alvo <= last_date.date():
        st.error("A data precisa ser futura (maior que a última data do CSV).")
        st.stop()

    alvo_ts = pd.to_datetime(alvo)
    horizon = int((alvo_ts - pd.to_datetime(last_date.date())).days)
    st.write(f"Dias simulados até a data alvo: **{horizon}**")

    mode = st.selectbox(
        "Cenário",
        ["Constante", "Constante + Ruído", "Aleatório (volatilidade)"],
        key="cenario_mode",
    )

    if mode == "Constante":
        mu = st.number_input("Retorno diário (%)", value=0.20, step=0.05, key="mu_const") / 100.0
        sigma = 0.0
    elif mode == "Constante + Ruído":
        mu = st.number_input("Retorno médio diário (%)", value=0.15, step=0.05, key="mu_ruido") / 100.0
        sigma = st.number_input("Ruído diário (%)", value=0.30, step=0.05, key="sigma_ruido") / 100.0
    else:
        mu = st.number_input("Retorno médio diário (%)", value=0.05, step=0.05, key="mu_alea") / 100.0
        sigma = st.number_input("Volatilidade diária (%)", value=0.80, step=0.05, key="sigma_alea") / 100.0

    seed = st.number_input("Seed", value=42, step=1, key="seed_sim")
    np.random.seed(int(seed))

    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
    rets = np.random.normal(loc=mu, scale=sigma, size=horizon)

    prices = [last_price]
    for r in rets:
        prices.append(prices[-1] * (1.0 + r))
    future_prices = prices[1:]

    base = df[["Data", "Vol.", "Último", "Abertura", "Máxima", "Mínima"]].copy()
    fut = pd.DataFrame({
        "Data": future_dates,
        "Vol.": [last_vol for _ in range(horizon)],
        "Último": future_prices,
        "Abertura": future_prices,
        "Máxima": [p * 1.002 for p in future_prices],
        "Mínima": [p * 0.998 for p in future_prices],
    })

    full = pd.concat([base, fut], ignore_index=True).sort_values("Data").reset_index(drop=True)
    full = correcao_escala_por_vizinhanca(full)
    full = compute_features_inplace(full)
    full = full.replace([np.inf, -np.inf], np.nan)

    start_future = last_date + timedelta(days=1)
    future_block_all = full[(full["Data"] >= start_future) & (full["Data"] <= alvo_ts)].copy()

    mask_valid = future_block_all[features].notna().all(axis=1)
    n_total = int(len(future_block_all))
    n_valid = int(mask_valid.sum())
    n_drop = n_total - n_valid

    st.caption(f"📌 Dias simulados: {n_total} | ✅ válidos: {n_valid} | 🧹 descartados: {n_drop}")

    if n_drop > 0:
        nan_counts = future_block_all.loc[~mask_valid, features].isna().sum().sort_values(ascending=False)
        nan_counts = nan_counts[nan_counts > 0].head(12)
        if len(nan_counts) > 0:
            with st.expander("Ver motivos do descarte (features com NaN/Inf)"):
                st.write(nan_counts)

    future_block = future_block_all.copy()
    future_block[features] = future_block[features].replace([np.inf, -np.inf], np.nan)
    future_block[features] = future_block[features].ffill().bfill().fillna(0.0)

    if len(future_block) == 0:
        st.warning("Não foi possível gerar dias simulados para a janela escolhida.")
        st.stop()

    Xf = future_block[features].values
    pred_f, proba_f = predict_proba_batch(model, scaler, Xf, threshold)

    future_block["P(ALTA)"] = proba_f
    future_block["Sinal"] = np.where(pred_f == 1, "ALTA", "BAIXA")

    if (future_block["Data"] == alvo_ts).any():
        row = future_block.loc[future_block["Data"] == alvo_ts].iloc[0]
    else:
        row = future_block.iloc[-1]

    proba_alvo = float(row["P(ALTA)"])
    data_real_alvo = pd.to_datetime(row["Data"]).date()
    sinal_alvo = int(proba_alvo >= threshold)

    append_usage_log({
        "action": "simulacao_futura",
        "status": "ok",
        "threshold": float(threshold),
        "alvo_user": str(alvo),
        "alvo_effective": str(data_real_alvo),
        "horizon_days": int(horizon),
        "mode": str(mode),
        "mu": float(mu),
        "sigma": float(sigma),
        "seed": int(seed),
        "n_total": n_total,
        "n_valid_original": n_valid,
        "n_drop_original": n_drop,
        "proba_alvo": float(proba_alvo),
        "pred_alvo": int(sinal_alvo),
    })

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data alvo (escolhida)", str(alvo))
    c2.metric("Data efetiva (prevista)", str(data_real_alvo))
    c3.metric("P(ALTA)", f"{proba_alvo:.2%}")
    c4.metric("Sinal", "ALTA" if sinal_alvo == 1 else "BAIXA")

    st.dataframe(future_block[["Data", "Último", "P(ALTA)", "Sinal"]], use_container_width=True)

    fig2 = make_signal_chart_intuitivo(
        df_plot=future_block,
        pred=(future_block["P(ALTA)"].values >= threshold).astype(int),
        proba=future_block["P(ALTA)"].values,
        threshold=threshold,
        title=f"Simulação futura — preço + probabilidade (até {alvo})",
        height=chart_height,
        show_rangeslider=show_rangeslider,
    )

    sim_key = f"sim_{alvo}_{mode}_{mu}_{sigma}_{seed}_{threshold}_{horizon}_{n_total}_{n_drop}"

    st.plotly_chart(
        fig2,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
        key=sim_key,
    )

# =========================
# TAB 2 — HISTÓRICO
# =========================
with tab_historico:
    st.subheader("Histórico: selecione uma data do dataset e obtenha a tendência do dia seguinte")
    st.info(
        "Aqui você trabalha com **dados reais do CSV**. Selecione uma data e veja a previsão do **dia seguinte** "
        f"como **P(ALTA)** e **Sinal** (usando o **Threshold** da lateral).",
        icon="ℹ️",
    )

    date_options = df["Data"].dt.date.tolist()
    selected_date = st.selectbox("Data (histórico)", options=date_options, index=len(date_options) - 1, key="hist_date")

    idx_list = df.index[df["Data"].dt.date == selected_date]
    idx = int(idx_list[0])

    X_sel = df.loc[[idx], features].values
    pred_sel, proba_sel = predict_proba_batch(model, scaler, X_sel, threshold)
    y = int(pred_sel[0])
    p = float(proba_sel[0])

    append_usage_log({
        "action": "historico_predicao",
        "threshold": float(threshold),
        "selected_date": str(selected_date),
        "proba": float(p),
        "pred": int(y),
    })

    if y == 1:
        st.success(f"📈 Tendência prevista (dia seguinte): **ALTA** — P(ALTA)={p:.2%}")
    else:
        st.warning(f"📉 Tendência prevista (dia seguinte): **BAIXA** — P(ALTA)={p:.2%}")

    cols_show = ["Data", "Último", "Vol.", "rsi", "macd", "bb_largura", "atr_pct", "Alvo"]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df.loc[[idx], cols_show], use_container_width=True)

    df_plot = df.tail(int(view_n)).copy()
    X_plot = df_plot[features].values
    pred_plot, proba_plot = predict_proba_batch(model, scaler, X_plot, threshold)

    fig = make_signal_chart_intuitivo(
        df_plot=df_plot,
        pred=pred_plot,
        proba=proba_plot,
        threshold=threshold,
        title="Histórico — preço + probabilidade",
        height=chart_height,
        show_rangeslider=show_rangeslider,
    )

    hist_key = f"hist_{selected_date}_{threshold}_{view_n}_{chart_height}_{show_rangeslider}"

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
        key=hist_key,
    )

# =========================
# TAB 3 — DIAGNÓSTICO
# =========================
with tab_diag:
    st.subheader("Painel explícito de métricas (fixas do Colab — sem re-treino)")
    st.info(
        "Este painel mostra as **métricas do treinamento no Colab** (fixas, sem re-treino aqui) "
        "e um resumo do período do dataset carregado.",
        icon="ℹ️",
    )

    st.caption(f"Modelo: {METRICAS_COLAB['modelo']}")
    st.caption(f"Validação: {METRICAS_COLAB['janela_validacao']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acurácia Treino", f"{METRICAS_COLAB['acc_train']*100:.2f}%")
    c2.metric("Acurácia Teste", f"{METRICAS_COLAB['acc_test']*100:.2f}%")
    c3.metric("Overfitting", f"{METRICAS_COLAB['overfit']*100:.2f}%")
    c4.metric("F1 (CV)", f"{METRICAS_COLAB['cv_f1_mean']:.3f} ± {METRICAS_COLAB['cv_f1_pm']:.3f}")

    st.divider()

    cm = np.array(METRICAS_COLAB["cm"], dtype=int)
    cm_df = pd.DataFrame(
        cm,
        index=["Real: Queda (0)", "Real: Alta (1)"],
        columns=["Prev: Queda (0)", "Prev: Alta (1)"]
    )
    st.dataframe(cm_df, use_container_width=True)

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=list(cm_df.columns),
        y=list(cm_df.index),
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
        hovertemplate="",
    ))
    fig_cm.update_layout(template="plotly_white", height=420, title="Matriz de Confusão (valores do Colab)")
    st.plotly_chart(fig_cm, use_container_width=True, config={"displaylogo": False})

    st.divider()
    st.write("Classification report (do Colab):")
    st.code(METRICAS_COLAB["report"])

    st.divider()
    d1, d2, d3 = st.columns(3)
    d1.metric("Linhas válidas (features)", len(df))
    d2.metric("Data inicial", str(df["Data"].iloc[0].date()))
    d3.metric("Data final", str(df["Data"].iloc[-1].date()))

    st.write("Resumo do `Último` (corrigido):")
    st.write(df["Último"].describe())

    st.write("Últimos 10 pontos (Data, Último):")
    st.dataframe(df[["Data", "Último"]].tail(10), use_container_width=True)

    append_usage_log({
        "action": "abrir_diagnostico",
        "threshold": float(threshold),
        "rows_features": int(len(df)),
        "date_min": str(df["Data"].iloc[0].date()),
        "date_max": str(df["Data"].iloc[-1].date()),
    })

# =========================
# ✅ TAB 4 — ENTRADA DE DADOS (UPLOAD / MANUAL)
# =========================
with tab_entrada:
    st.subheader("Entrada de Dados: Upload de CSV ou Entrada Manual (OHLCV)")
    st.info(
        "Aqui você pode **inserir seus próprios dados** de duas formas:\n"
        "1) **Upload de CSV** com histórico\n"
        "2) **Entrada manual** de um dia (OHLCV) adicionada ao histórico do app\n\n"
        "As abas anteriores permanecem iguais.",
        icon="ℹ️",
    )

    modo = st.radio("Modo de entrada", ["📤 Upload de CSV (histórico)", "✍️ Entrada manual (um dia OHLCV)"], horizontal=True)

    # ---------- UPLOAD CSV ----------
    if modo.startswith("📤"):
        st.markdown("### 1) Upload de CSV (histórico)")
        st.caption("O CSV deve conter as colunas: **Data, Último, Abertura, Máxima, Mínima, Vol.**")

        up = st.file_uploader("Envie seu CSV", type=["csv"], accept_multiple_files=False)

        if up is None:
            st.warning("Envie um CSV para carregar seus dados.")
            st.stop()

        try:
            df_u = carregar_dados_upload(up)
            features_u = df_u.attrs["features_sugeridas"]
        except Exception as e:
            st.error(f"Não consegui processar o CSV enviado: {e}")
            st.stop()

        append_usage_log({
            "action": "upload_csv",
            "status": "ok",
            "filename": getattr(up, "name", "uploaded.csv"),
            "rows_valid": int(len(df_u)),
        })

        st.success(f"CSV carregado: **{getattr(up, 'name', 'uploaded.csv')}** | Linhas válidas: **{len(df_u)}**")
        st.caption(f"Período: {df_u['Data'].iloc[0].date()} → {df_u['Data'].iloc[-1].date()}")

        # Previsão por data (igual ao Histórico, mas usando o CSV do usuário)
        date_options_u = df_u["Data"].dt.date.tolist()
        selected_date_u = st.selectbox(
            "Selecione uma data do seu CSV (previsão para o dia seguinte)",
            options=date_options_u,
            index=len(date_options_u) - 1,
            key="upload_hist_date",
        )

        idx_list_u = df_u.index[df_u["Data"].dt.date == selected_date_u]
        idx_u = int(idx_list_u[0])

        X_sel_u = df_u.loc[[idx_u], features_u].values
        pred_sel_u, proba_sel_u = predict_proba_batch(model, scaler, X_sel_u, threshold)
        y_u = int(pred_sel_u[0])
        p_u = float(proba_sel_u[0])

        append_usage_log({
            "action": "upload_predicao_data",
            "threshold": float(threshold),
            "selected_date": str(selected_date_u),
            "proba": float(p_u),
            "pred": int(y_u),
        })

        if y_u == 1:
            st.success(f"📈 Tendência prevista (dia seguinte): **ALTA** — P(ALTA)={p_u:.2%}")
        else:
            st.warning(f"📉 Tendência prevista (dia seguinte): **BAIXA** — P(ALTA)={p_u:.2%}")

        cols_show_u = ["Data", "Último", "Vol.", "rsi", "macd", "bb_largura", "atr_pct", "Alvo"]
        cols_show_u = [c for c in cols_show_u if c in df_u.columns]
        st.dataframe(df_u.loc[[idx_u], cols_show_u], use_container_width=True)

        # Gráfico do CSV do usuário (últimos N)
        df_plot_u = df_u.tail(int(view_n)).copy()
        X_plot_u = df_plot_u[features_u].values
        pred_plot_u, proba_plot_u = predict_proba_batch(model, scaler, X_plot_u, threshold)

        fig_u = make_signal_chart_intuitivo(
            df_plot=df_plot_u,
            pred=pred_plot_u,
            proba=proba_plot_u,
            threshold=threshold,
            title="Upload CSV — preço + probabilidade",
            height=chart_height,
            show_rangeslider=show_rangeslider,
        )

        upload_key = f"upload_{getattr(up,'name','csv')}_{selected_date_u}_{threshold}_{view_n}_{chart_height}_{show_rangeslider}"
        st.plotly_chart(
            fig_u,
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
            key=upload_key,
        )

    # ---------- ENTRADA MANUAL ----------
    else:
        st.markdown("### 2) Entrada manual (um dia OHLCV)")
        st.caption(
            "Você informa **um dia** (OHLCV) e o app **anexa ao histórico atual** para conseguir calcular as features "
            "(RSI/MACD/Bollinger/ATR etc.). Depois ele prevê a tendência do **dia seguinte** a esse dia inserido."
        )

        last_date = pd.to_datetime(df["Data"].iloc[-1])
        last_price = float(df["Último"].iloc[-1])
        last_vol = float(df["Vol."].iloc[-1]) if pd.notna(df["Vol."].iloc[-1]) else 0.0

        st.info(f"Última data do histórico do app: **{last_date.date()}**", icon="ℹ️")

        with st.form("manual_form", clear_on_submit=False):
            data_manual = st.date_input(
                "Data do registro manual (precisa ser > última data do histórico)",
                value=(last_date + timedelta(days=1)).date(),
                key="manual_date",
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                ultimo = st.number_input("Último", value=float(last_price), step=1.0, key="m_ultimo")
                abertura = st.number_input("Abertura", value=float(last_price), step=1.0, key="m_abertura")
            with c2:
                maxima = st.number_input("Máxima", value=float(last_price) * 1.01, step=1.0, key="m_maxima")
                minima = st.number_input("Mínima", value=float(last_price) * 0.99, step=1.0, key="m_minima")
            with c3:
                vol = st.number_input("Vol. (numérico)", value=float(last_vol) if last_vol > 0 else 1_000_000.0, step=1000.0, key="m_vol")

            submitted = st.form_submit_button("Calcular previsão para o dia seguinte")

        if not submitted:
            st.stop()

        if data_manual <= last_date.date():
            st.error("A data manual precisa ser **maior** que a última data do histórico.")
            st.stop()

        # Monta uma linha manual e anexa ao histórico do app (sem mexer nas abas atuais)
        manual_row = pd.DataFrame([{
            "Data": pd.to_datetime(data_manual),
            "Vol.": float(vol),
            "Último": float(ultimo),
            "Abertura": float(abertura),
            "Máxima": float(maxima),
            "Mínima": float(minima),
        }])

        base = df[["Data", "Vol.", "Último", "Abertura", "Máxima", "Mínima"]].copy()
        full_m = pd.concat([base, manual_row], ignore_index=True).sort_values("Data").reset_index(drop=True)
        full_m = correcao_escala_por_vizinhanca(full_m)
        full_m = compute_features_inplace(full_m)
        full_m = full_m.replace([np.inf, -np.inf], np.nan)

        # pega exatamente a linha manual (última por data, já que data_manual > last_date)
        row_m = full_m.iloc[-1:].copy()

        # se ainda tiver NaN em features (caso histórico curto), imputa para não quebrar
        row_m[features] = row_m[features].replace([np.inf, -np.inf], np.nan)
        row_m[features] = row_m[features].ffill().bfill().fillna(0.0)

        Xm = row_m[features].values
        pred_m, proba_m = predict_proba_batch(model, scaler, Xm, threshold)

        y_m = int(pred_m[0])
        p_m = float(proba_m[0])

        append_usage_log({
            "action": "manual_ohlcv",
            "status": "ok",
            "threshold": float(threshold),
            "data_manual": str(data_manual),
            "ultimo": float(ultimo),
            "abertura": float(abertura),
            "maxima": float(maxima),
            "minima": float(minima),
            "vol": float(vol),
            "proba": float(p_m),
            "pred": int(y_m),
        })

        cA, cB, cC = st.columns(3)
        cA.metric("Data manual", str(data_manual))
        cB.metric("P(ALTA) (dia seguinte)", f"{p_m:.2%}")
        cC.metric("Sinal", "ALTA" if y_m == 1 else "BAIXA")

        # Mostra a linha manual (com algumas features)
        cols_show_m = ["Data", "Último", "Vol.", "rsi", "macd", "bb_largura", "atr_pct"]
        cols_show_m = [c for c in cols_show_m if c in full_m.columns]
        st.dataframe(full_m.tail(1)[cols_show_m], use_container_width=True)

        # Gráfico: últimos N do histórico + ponto manual, com probabilidade/sinal recalculados
        df_plot_m = full_m.tail(int(view_n)).copy()
        df_plot_m[features] = df_plot_m[features].replace([np.inf, -np.inf], np.nan)
        df_plot_m[features] = df_plot_m[features].ffill().bfill().fillna(0.0)

        X_plot_m = df_plot_m[features].values
        pred_plot_m, proba_plot_m = predict_proba_batch(model, scaler, X_plot_m, threshold)

        fig_m = make_signal_chart_intuitivo(
            df_plot=df_plot_m.assign(**{"P(ALTA)": proba_plot_m}),
            pred=pred_plot_m,
            proba=proba_plot_m,
            threshold=threshold,
            title="Entrada manual — histórico + ponto inserido (preço + probabilidade)",
            height=chart_height,
            show_rangeslider=show_rangeslider,
        )

        manual_key = f"manual_{data_manual}_{threshold}_{view_n}_{chart_height}_{show_rangeslider}"
        st.plotly_chart(
            fig_m,
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
            key=manual_key,
        )
