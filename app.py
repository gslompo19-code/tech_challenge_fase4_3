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
# Funções do seu notebook
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
    """
    Correção de escala baseada na vizinhança temporal (igual ao seu ajuste do Colab).
    """
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

    # Correção de escala (patch do colab)
    df = corrige_escala_ultimo(df)
    df = df.sort_values("Data").dropna(subset=["Data", "Último"]).reset_index(drop=True)

    # Features base
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


# =========================
# Modelo / scaler / log
# =========================
def load_scaler_or_fit(X_train_raw):
    if os.path.exists(SCALER_PATH):
        try:
            return joblib.load(SCALER_PATH)
        except:
            pass
    return MinMaxScaler().fit(X_train_raw)


def load_model_or_train(X_train, y_train):
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except:
            pass

    model = CatBoostClassifier(
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
    model.fit(X_train, y_train)
    return model


def ensure_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    header = [
        "timestamp", "source", "action",
        "selected_date", "pred_direction", "pred_proba",
        "threshold"
    ]
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
# Simulação futura por cenário
# =========================
def simular_futuro_cenario(df_base: pd.DataFrame, dias_a_frente: int, retorno_diario: float):
    df_sim = df_base.copy()

    last_date = df_sim["Data"].iloc[-1]
    last_close = float(df_sim["Último"].iloc[-1])

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=dias_a_frente, freq="D")

    future_close = []
    price = last_close
    for _ in range(dias_a_frente):
        price = price * (1 + retorno_diario)
        future_close.append(price)

    future_df = pd.DataFrame({
        "Data": future_dates,
        "Último": future_close,
        "Abertura": future_close,
        "Máxima": future_close,
        "Mínima": future_close,
        "Vol.": [float(df_sim["Vol."].iloc[-1])] * dias_a_frente,
    })

    df_all = pd.concat([df_sim, future_df], ignore_index=True)
    df_all = df_all.sort_values("Data").reset_index(drop=True)

    # recomputa features (mesma lógica do histórico)
    df_all["var_pct"] = df_all["Último"].pct_change()
    for dias in [3, 7, 14, 21, 30]:
        df_all[f"mm_{dias}"] = df_all["Último"].rolling(dias, min_periods=dias).mean()
    for dias in [5, 10, 20]:
        df_all[f"vol_{dias}"] = df_all["Último"].rolling(dias, min_periods=dias).std()

    df_all["desvio_mm3"] = df_all["Último"] - df_all["mm_3"]
    df_all["dia"] = df_all["Data"].dt.weekday
    df_all["rsi"] = calculate_rsi(df_all["Último"])

    macd, sinal, hist = macd_components(df_all["Último"])
    df_all["macd"], df_all["sinal_macd"], df_all["hist_macd"] = macd, sinal, hist

    bb_media = df_all["Último"].rolling(20, min_periods=20).mean()
    bb_std = df_all["Último"].rolling(20, min_periods=20).std()
    df_all["bb_media"] = bb_media
    df_all["bb_std"] = bb_std
    df_all["bb_sup"] = bb_media + 2 * bb_std
    df_all["bb_inf"] = bb_media - 2 * bb_std
    df_all["bb_largura"] = (df_all["bb_sup"] - df_all["bb_inf"]) / bb_media

    tr1 = df_all["Máxima"] - df_all["Mínima"]
    tr2 = (df_all["Máxima"] - df_all["Último"].shift(1)).abs()
    tr3 = (df_all["Mínima"] - df_all["Último"].shift(1)).abs()
    df_all["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_all["ATR"] = df_all["TR"].rolling(14, min_periods=14).mean()

    df_all["obv"] = obv_series(df_all)

    df_all["ret_1d"] = df_all["Último"].pct_change()
    df_all["log_ret"] = np.log(df_all["Último"]).diff()
    df_all["ret_5d"] = df_all["Último"].pct_change(5)
    df_all["rv_20"] = df_all["ret_1d"].rolling(20, min_periods=20).std()

    df_all["atr_pct"] = df_all["ATR"] / df_all["Último"]
    df_all["desvio_mm3_pct"] = (df_all["desvio_mm3"] / df_all["mm_3"]).replace([np.inf, -np.inf], np.nan)

    df_all["vol_log"] = np.log(df_all["Vol."].clip(lower=1))
    df_all["vol_ret"] = df_all["Vol."].pct_change().replace([np.inf, -np.inf], np.nan)

    df_all["obv_diff"] = pd.Series(df_all["obv"]).diff()

    df_all["z_close_20"] = zscore_roll(df_all["Último"], 20)
    df_all["z_rsi_20"] = zscore_roll(df_all["rsi"], 20)
    df_all["z_macd_20"] = zscore_roll(df_all["macd"], 20)

    features = df_base.attrs.get("features_sugeridas", [])
    df_future = df_all.tail(dias_a_frente).copy()
    return df_future, features


# =========================
# Sidebar
# =========================
st.title("📈 IBOV Signal — Sistema Preditivo (Interativo)")
st.caption("Produto: escolha uma data e veja tendência prevista. Simulação: gere sinais futuros por cenário.")

with st.sidebar:
    st.header("Dados")
    uploaded = st.file_uploader("Upload de CSV (opcional)", type=["csv"])

    test_n = st.number_input("Janela de teste (últimos N)", min_value=10, max_value=260, value=60, step=10)

    st.header("Decisão")
    threshold = st.slider("Threshold P(ALTA) ≥ t", 0.30, 0.70, 0.50, 0.01)

    st.header("Simulação futura")
    fut_days = st.number_input("Dias à frente (até 30)", min_value=1, max_value=30, value=30, step=1)
    fut_ret = st.slider("Retorno diário do cenário (%)", -2.0, 2.0, 0.2, 0.1)

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

# scaler/model
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
    except:
        scaler = MinMaxScaler().fit(X_train_raw)
else:
    scaler = MinMaxScaler().fit(X_train_raw)

X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except:
        model = load_model_or_train(X_train, y_train)
else:
    model = load_model_or_train(X_train, y_train)

# probas/sinais histórico
X_all_scaled = scaler.transform(X_raw)
if hasattr(model, "predict_proba"):
    proba_all = model.predict_proba(X_all_scaled)[:, 1]
else:
    proba_all = model.predict(X_all_scaled).astype(float)

pred_all = (proba_all >= threshold).astype(int)


# =========================
# Tabs
# =========================
tab_prod, tab_analises, tab_future, tab_about = st.tabs(
    ["🧠 Produto (interativo)", "📊 Análises Temporais", "🔮 Simulação Futura", "📘 Sobre"]
)

# ======================================================
# TAB 1: PRODUTO INTERATIVO
# ======================================================
with tab_prod:
    st.subheader("Selecione uma data e veja a tendência para o dia seguinte")

    date_options = df["Data"].dt.date.tolist()
    selected_date = st.selectbox("Data (histórico)", options=date_options, index=len(date_options) - 1)

    idx_list = df.index[df["Data"].dt.date == selected_date]
    if len(idx_list) == 0:
        st.error("Data não encontrada.")
        st.stop()
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

    # gráfico mais intuitivo
    with st.expander("Ajustes do gráfico", expanded=True):
        view_n = st.slider("Mostrar últimos N pontos", 60, min(1500, len(df)), 400, 20)

    df_plot = df.tail(int(view_n)).copy()
    start_idx_plot = len(df) - len(df_plot)

    proba_plot = proba_all[start_idx_plot:start_idx_plot + len(df_plot)]
    pred_plot = pred_all[start_idx_plot:start_idx_plot + len(df_plot)]
    price_vals = df_plot["Último"].astype(float).values

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=("Preço + Sinal", "Probabilidade de ALTA (com threshold)")
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot["Data"], y=df_plot["Último"],
            mode="lines", name="Preço (Último)",
            hovertemplate="Data=%{x|%Y-%m-%d}<br>Preço=%{y:.2f}<extra></extra>",
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot["Data"],
            y=np.where(pred_plot == 1, price_vals, np.nan),
            mode="markers", name="Sinal: ALTA",
            marker=dict(size=9, symbol="triangle-up"),
            hovertemplate="Data=%{x|%Y-%m-%d}<br>Sinal=ALTA<br>Preço=%{y:.2f}<extra></extra>",
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot["Data"],
            y=np.where(pred_plot == 0, price_vals, np.nan),
            mode="markers", name="Sinal: BAIXA",
            marker=dict(size=8, symbol="triangle-down"),
            hovertemplate="Data=%{x|%Y-%m-%d}<br>Sinal=BAIXA<br>Preço=%{y:.2f}<extra></extra>",
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot["Data"], y=proba_plot,
            mode="lines", fill="tozeroy", name="P(ALTA)",
            hovertemplate="Data=%{x|%Y-%m-%d}<br>P(ALTA)=%{y:.2%}<extra></extra>",
        ),
        row=2, col=1
    )

    fig.add_hline(y=threshold, line_width=2, line_dash="dash", annotation_text=f"threshold={threshold:.2f}", row=2, col=1)

    # fundo por blocos de sinal
    blocks = []
    curr = int(pred_plot[0])
    block_start = df_plot["Data"].iloc[0]
    for k in range(1, len(df_plot)):
        if int(pred_plot[k]) != curr:
            blocks.append((block_start, df_plot["Data"].iloc[k - 1], curr))
            curr = int(pred_plot[k])
            block_start = df_plot["Data"].iloc[k]
    blocks.append((block_start, df_plot["Data"].iloc[-1], curr))

    for (x0, x1, s) in blocks:
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="rgba(0,180,0,0.06)" if s == 1 else "rgba(220,0,0,0.05)",
            line_width=0,
            row=1, col=1
        )

    fig.add_vline(x=pd.to_datetime(selected_date), line_width=2, row=1, col=1)
    fig.add_vline(x=pd.to_datetime(selected_date), line_width=2, row=2, col=1)

    fig.update_layout(height=650, margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h"))
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="P(ALTA)", range=[0, 1], row=2, col=1)
    fig.update_xaxes(rangeslider_visible=True)

    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Preço na data", f"{float(df.loc[i,'Último']):,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    c2.metric("P(ALTA)", f"{p_sel:.2%}")
    c3.metric("Sinal", "ALTA" if y_sel == 1 else "BAIXA")


# ======================================================
# TAB 2: ANÁLISES TEMPORAIS (sem buy&hold)
# ======================================================
with tab_analises:
    st.subheader("Qualidade do modelo no período de teste (últimos N)")

    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia (teste)", f"{acc:.2%}")
    c2.metric("F1-score (teste)", f"{f1:.3f}")
    c3.metric("Janela teste", f"{int(test_n)} dias")

    st.write("Matriz de confusão:")
    st.write(confusion_matrix(y_test, y_pred_test))

    st.text("Relatório:")
    st.text(classification_report(y_test, y_pred_test))

    st.divider()
    st.subheader("Rolling (janela móvel) — estabilidade no tempo")

    rw = st.slider("Janela rolling", 20, min(200, len(y_test)), 60, 10)
    test_dates = df["Data"].iloc[-len(y_test):].reset_index(drop=True)

    roll_acc = [np.nan] * len(y_test)
    roll_f1 = [np.nan] * len(y_test)

    for k in range(rw - 1, len(y_test)):
        yt = y_test[k - rw + 1:k + 1]
        yp = y_pred_test[k - rw + 1:k + 1]
        roll_acc[k] = accuracy_score(yt, yp)
        roll_f1[k] = f1_score(yt, yp)

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=test_dates, y=roll_acc, mode="lines", name=f"Acurácia rolling ({rw})"))
    fig_r.add_trace(go.Scatter(x=test_dates, y=roll_f1, mode="lines", name=f"F1 rolling ({rw})"))
    fig_r.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_r, use_container_width=True)


# ======================================================
# TAB 3: SIMULAÇÃO FUTURA (30 dias + tendência vs dia anterior)
# ======================================================
with tab_future:
    st.subheader("Simulação futura (cenário) — tendência vs dia anterior")

    st.caption(
        "Aqui a tendência **ALTA/BAIXA** é calculada no futuro como: "
        "**Preço_simulado(d) > Preço_simulado(d-1)**. "
        "Além disso, exibimos a **previsão do modelo** (P(ALTA)) para cada dia simulado."
    )

    retorno_diario = float(fut_ret) / 100.0
    df_future, fcols = simular_futuro_cenario(df, int(fut_days), retorno_diario)

    X_future_raw = df_future[fcols].values
    if np.isnan(X_future_raw).any():
        st.warning("Algumas features ficaram NaN (por janelas). Diminua dias ou aumente histórico.")
        st.stop()

    X_future = scaler.transform(X_future_raw)

    if hasattr(model, "predict_proba"):
        future_proba = model.predict_proba(X_future)[:, 1]
    else:
        future_proba = model.predict(X_future).astype(float)

    future_pred = (future_proba >= threshold).astype(int)

    out = pd.DataFrame({
        "Data": df_future["Data"].dt.date,
        "Preço Simulado": df_future["Último"].astype(float),
        "P(ALTA)": future_proba.astype(float),
        "Sinal Modelo": np.where(future_pred == 1, "ALTA", "BAIXA"),
    })

    # tendência real do cenário vs dia anterior (do próprio cenário simulado)
    out["Variação vs ontem"] = out["Preço Simulado"].diff()
    out["Tendência Cenário"] = np.where(out["Variação vs ontem"] > 0, "ALTA", "BAIXA")
    out.loc[out.index[0], "Tendência Cenário"] = "—"  # primeiro dia não tem anterior

    append_log(
        source_name,
        "future_scenario_30d",
        str(df["Data"].iloc[-1].date()),
        int(future_pred[-1]),
        float(future_proba[-1]),
        threshold
    )

    # --------- Gráfico intuitivo: preço + setas + prob embaixo
    figf = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=("Preço simulado + tendência vs ontem", "P(ALTA) do modelo (com threshold)")
    )

    # linha do preço
    figf.add_trace(
        go.Scatter(
            x=out["Data"], y=out["Preço Simulado"],
            mode="lines+markers",
            name="Preço Simulado",
            hovertemplate="Data=%{x}<br>Preço=%{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    # setas por tendência vs ontem
    prices = out["Preço Simulado"].values
    up_y = np.where(out["Variação vs ontem"].values > 0, prices, np.nan)
    dn_y = np.where(out["Variação vs ontem"].values <= 0, prices, np.nan)

    figf.add_trace(
        go.Scatter(
            x=out["Data"], y=up_y,
            mode="markers",
            name="Tendência vs ontem: ALTA",
            marker=dict(size=11, symbol="triangle-up"),
            hovertemplate="Data=%{x}<br>Tendência=ALTA<br>Preço=%{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    figf.add_trace(
        go.Scatter(
            x=out["Data"], y=dn_y,
            mode="markers",
            name="Tendência vs ontem: BAIXA",
            marker=dict(size=10, symbol="triangle-down"),
            hovertemplate="Data=%{x}<br>Tendência=BAIXA<br>Preço=%{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    # prob como área (modelo)
    figf.add_trace(
        go.Scatter(
            x=out["Data"], y=out["P(ALTA)"],
            mode="lines",
            fill="tozeroy",
            name="P(ALTA) - Modelo",
            hovertemplate="Data=%{x}<br>P(ALTA)=%{y:.2%}<extra></extra>"
        ),
        row=2, col=1
    )

    # threshold
    figf.add_hline(y=threshold, line_dash="dash", line_width=2, annotation_text=f"threshold={threshold:.2f}", row=2, col=1)

    figf.update_layout(height=650, margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h"))
    figf.update_yaxes(title_text="Preço Simulado", row=1, col=1)
    figf.update_yaxes(title_text="P(ALTA)", range=[0, 1], row=2, col=1)
    figf.update_xaxes(rangeslider_visible=True)

    st.plotly_chart(figf, use_container_width=True)

    st.subheader("Tabela da simulação (30 dias)")
    st.dataframe(out, use_container_width=True)


# ======================================================
# TAB 4: SOBRE
# ======================================================
with tab_about:
    st.subheader("O que o modelo prevê")
    st.write("**Alvo:** 1 se `Último(t+1) > Último(t)`, senão 0.")
    st.write("**No produto:** usuário escolhe uma data e o app retorna ALTA/BAIXA do próximo dia com P(ALTA).")

    st.subheader("Simulação futura")
    st.write(
        "Como não existe histórico real para datas futuras, usamos um **cenário** (retorno diário) "
        "para construir um caminho de preços e calcular features. "
        "O gráfico mostra duas coisas:\n"
        "- **Tendência do cenário** (preço hoje vs ontem)\n"
        "- **Sinal do modelo** (probabilidade e decisão por threshold)"
    )

    st.subheader("Prevenção de vazamento")
    st.write("- Split temporal (treino antes, teste nos últimos N)\n- Scaler com fit apenas no treino")

    st.subheader("Features usadas")
    st.write(features)

    st.subheader("Arquivos recomendados no repositório")
    st.write(
        "- `README.md` (como rodar + link do app)\n"
        "- `MODEL_CARD.md` (estratégia, validação, limitações)\n"
        "- `modelo_catboost.pkl` e `scaler_minmax.pkl` (artefatos)\n"
        "- `Dados Ibovespa (2).csv` (dataset)"
    )


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
