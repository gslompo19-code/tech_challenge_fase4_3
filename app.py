import os
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import joblib


# =========================
# Config
# =========================
st.set_page_config(page_title="IBOV Signal — Sistema Preditivo", layout="wide")

DEFAULT_CSV = "Dados Ibovespa (2).csv"
MODEL_PATH = "modelo_catboost.pkl"
SCALER_PATH = "scaler_minmax.pkl"


# =========================
# MÉTRICAS FIXAS (do Colab) — SEM RETREINO
# =========================
METRICAS_COLAB = {
    "modelo": "CatBoostClassifier (treinado no Colab / Fase 2)",
    "janela_validacao": "Holdout temporal: últimos 30 registros como teste",
    "cv_f1_mean": 0.531,
    "cv_f1_pm": 0.083,  # (+/- 0.083) conforme seu print
    "acc_train": 0.8203,
    "acc_test": 0.8000,
    "overfit": 0.0203,
    "cm": [[13, 3],
           [3, 11]],  # Confusão do seu relatório do Colab
    "report": """precision    recall  f1-score   support

0       0.81      0.81      0.81        16
1       0.79      0.79      0.79        14

accuracy                           0.80        30
macro avg       0.80      0.80      0.80        30
weighted avg    0.80      0.80      0.80        30"""
}


# =========================
# Funções do Colab
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


def correcao_escala_por_vizinhanca(df: pd.DataFrame) -> pd.DataFrame:
    """
    Patch idêntico ao seu Colab para corrigir 'Último' quando vem 10x/100x/1000x menor.
    """
    df = df.copy()

    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df["Último"] = pd.to_numeric(df["Último"], errors="coerce")

    df = df.dropna(subset=["Data", "Último"]).sort_values("Data").reset_index(drop=True)

    for i in range(1, len(df)):
        prev = df.loc[i - 1, "Último"]
        curr = df.loc[i, "Último"]

        if curr < prev * 0.2:
            for fator in [10, 100, 1000]:
                if prev * 0.7 < curr * fator < prev * 1.3:
                    df.loc[i, "Último"] = curr * fator
                    break

    return df


def carregar_dados(caminho_csv):
    """
    Pipeline do Colab + patch de correção de escala (para evitar gráfico 'pente').
    """
    df = pd.read_csv(caminho_csv)
    df.columns = df.columns.str.strip()
    df["Data"] = pd.to_datetime(df["Data"], format="%d.%m.%Y", errors="coerce")
    df = df.sort_values("Data").dropna(subset=["Data"])

    df["Vol."] = df["Vol."].apply(volume_to_float)

    for coluna in ["Último", "Abertura", "Máxima", "Mínima"]:
        df[coluna] = (
            df[coluna].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    # PATCH (igual Colab)
    df = correcao_escala_por_vizinhanca(df)
    df = df.sort_values("Data").reset_index(drop=True)

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
# Cache de carga
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
# Gráficos
# =========================
def make_signal_chart(df_plot, pred, proba, threshold, title):
    price_vals = df_plot["Último"].astype(float).values
    dates = df_plot["Data"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=("Preço (corrigido) + Sinal", "Probabilidade P(ALTA)")
    )

    fig.add_trace(
        go.Scatter(x=dates, y=price_vals, mode="lines", name="Preço (Último)"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dates, y=np.where(pred == 1, price_vals, np.nan),
            mode="markers", name="ALTA", marker=dict(size=9, symbol="triangle-up")
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=np.where(pred == 0, price_vals, np.nan),
            mode="markers", name="BAIXA", marker=dict(size=8, symbol="triangle-down")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=dates, y=proba, mode="lines", fill="tozeroy", name="P(ALTA)"),
        row=2, col=1
    )
    fig.add_hline(y=threshold, line_dash="dash", line_width=2, annotation_text=f"threshold={threshold:.2f}", row=2, col=1)

    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h"),
        title=title,
    )
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="P(ALTA)", range=[0, 1], row=2, col=1)
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def predict_proba_batch(model, scaler, X, threshold):
    Xs = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)[:, 1]
    else:
        proba = model.predict(Xs).astype(float)
    pred = (proba >= threshold).astype(int)
    return pred, proba


def plot_confusion_matrix(cm, labels=("Queda (0)", "Alta (1)")):
    cm = np.array(cm, dtype=int)
    x = [f"Prev: {labels[0]}", f"Prev: {labels[1]}"]
    y = [f"Real: {labels[0]}", f"Real: {labels[1]}"]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=x,
            y=y,
            text=cm,
            texttemplate="%{text}",
            hovertemplate="",
        )
    )
    fig.update_layout(
        title="Matriz de Confusão (valores do Colab)",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


# =========================
# App
# =========================
st.title("📈 IBOV Signal — Sistema Preditivo (modelo do Colab, sem re-treino)")

with st.sidebar:
    st.header("Config")
    threshold = st.slider("Threshold para ALTA", 0.30, 0.70, 0.50, 0.01)
    view_n = st.slider("Janela do gráfico (últimos N)", 60, 1500, 400, 20)
    st.caption("Patch de escala do `Último` aplicado (corrige gráfico 'pente').")

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

tab_produto, tab_futuro, tab_diag = st.tabs(["🧠 Produto", "🔮 Simulação futura (data manual)", "🔎 Diagnóstico (métricas)"])

with tab_produto:
    st.subheader("Produto: selecione uma data e obtenha a tendência do dia seguinte")

    date_options = df["Data"].dt.date.tolist()
    selected_date = st.selectbox("Data (histórico)", options=date_options, index=len(date_options) - 1)

    idx_list = df.index[df["Data"].dt.date == selected_date]
    idx = int(idx_list[0])

    X_sel = df.loc[[idx], features].values
    pred_sel, proba_sel = predict_proba_batch(model, scaler, X_sel, threshold)
    y = int(pred_sel[0])
    p = float(proba_sel[0])

    if y == 1:
        st.success(f"📈 Tendência prevista (dia seguinte): **ALTA** — P(ALTA)={p:.2%}")
    else:
        st.warning(f"📉 Tendência prevista (dia seguinte): **BAIXA** — P(ALTA)={p:.2%}")

    st.write("Linha do dia selecionado (já com `Último` corrigido):")
    st.dataframe(df.loc[[idx], ["Data", "Último", "Vol.", "rsi", "macd", "bb_largura", "atr_pct", "Alvo"]], use_container_width=True)

    df_plot = df.tail(int(view_n)).copy()
    X_plot = df_plot[features].values
    pred_plot, proba_plot = predict_proba_batch(model, scaler, X_plot, threshold)

    fig = make_signal_chart(df_plot, pred_plot, proba_plot, threshold, "Histórico + Sinais do modelo (com correção de escala)")
    st.plotly_chart(fig, use_container_width=True)

with tab_futuro:
    st.subheader("Simulação futura (cenário) até uma data escolhida")
    st.write(
        "Como não existe preço real futuro no dataset, aqui é uma **simulação de cenário**: "
        "você define um retorno diário e o modelo classifica ALTA/BAIXA para cada dia simulado."
    )

    last_date = pd.to_datetime(df["Data"].iloc[-1])
    last_price = float(df["Último"].iloc[-1])
    last_vol = float(df["Vol."].iloc[-1]) if pd.notna(df["Vol."].iloc[-1]) else 0.0

    st.info(
        f"Último ponto: {last_date.date()} — Último={last_price:,.2f}"
        .replace(",", "X").replace(".", ",").replace("X", ".")
    )

    alvo = st.date_input("Digite/Selecione a data futura", value=(last_date + timedelta(days=30)).date())
    if alvo <= last_date.date():
        st.error("A data precisa ser futura (maior que a última data do CSV).")
        st.stop()

    horizon = int((pd.to_datetime(alvo) - pd.to_datetime(last_date.date())).days)
    st.write(f"Dias simulados até a data alvo: **{horizon}**")

    mode = st.selectbox("Cenário", ["Constante", "Constante + Ruído", "Aleatório (volatilidade)"])
    if mode == "Constante":
        mu = st.number_input("Retorno diário (%)", value=0.20, step=0.05) / 100.0
        sigma = 0.0
    elif mode == "Constante + Ruído":
        mu = st.number_input("Retorno médio diário (%)", value=0.15, step=0.05) / 100.0
        sigma = st.number_input("Ruído diário (%)", value=0.30, step=0.05) / 100.0
    else:
        mu = st.number_input("Retorno médio diário (%)", value=0.05, step=0.05) / 100.0
        sigma = st.number_input("Volatilidade diária (%)", value=0.80, step=0.05) / 100.0

    seed = st.number_input("Seed", value=42, step=1)
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

    # Recalcular features no full
    full["var_pct"] = full["Último"].pct_change()
    for dias in [3, 7, 14, 21, 30]:
        full[f"mm_{dias}"] = full["Último"].rolling(dias, min_periods=dias).mean()
    for dias in [5, 10, 20]:
        full[f"vol_{dias}"] = full["Último"].rolling(dias, min_periods=dias).std()

    full["desvio_mm3"] = full["Último"] - full["mm_3"]
    full["dia"] = pd.to_datetime(full["Data"]).dt.weekday
    full["rsi"] = calculate_rsi(full["Último"])

    macd, sinal, hist = macd_components(full["Último"])
    full["macd"], full["sinal_macd"], full["hist_macd"] = macd, sinal, hist

    bb_media = full["Último"].rolling(20, min_periods=20).mean()
    bb_std = full["Último"].rolling(20, min_periods=20).std()
    full["bb_media"] = bb_media
    full["bb_std"] = bb_std
    full["bb_sup"] = bb_media + 2 * bb_std
    full["bb_inf"] = bb_media - 2 * bb_std
    full["bb_largura"] = (full["bb_sup"] - full["bb_inf"]) / bb_media

    tr1 = full["Máxima"] - full["Mínima"]
    tr2 = (full["Máxima"] - full["Último"].shift(1)).abs()
    tr3 = (full["Mínima"] - full["Último"].shift(1)).abs()
    full["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    full["ATR"] = full["TR"].rolling(14, min_periods=14).mean()

    full["obv"] = obv_series(full)

    full["ret_1d"] = full["Último"].pct_change()
    full["log_ret"] = np.log(full["Último"]).diff()
    full["ret_5d"] = full["Último"].pct_change(5)
    full["rv_20"] = full["ret_1d"].rolling(20, min_periods=20).std()

    full["atr_pct"] = full["ATR"] / full["Último"]
    full["desvio_mm3_pct"] = (full["desvio_mm3"] / full["mm_3"]).replace([np.inf, -np.inf], np.nan)

    full["vol_log"] = np.log(full["Vol."].clip(lower=1))
    full["vol_ret"] = full["Vol."].pct_change().replace([np.inf, -np.inf], np.nan)

    full["obv_diff"] = pd.Series(full["obv"]).diff()

    full["z_close_20"] = zscore_roll(full["Último"], 20)
    full["z_rsi_20"] = zscore_roll(full["rsi"], 20)
    full["z_macd_20"] = zscore_roll(full["macd"], 20)

    future_block = full[full["Data"].isin(future_dates)].copy()
    future_block = future_block.dropna(subset=features)

    if len(future_block) == 0:
        st.error("Sem features suficientes. A data precisa estar mais distante (janelas 20/30) ou o cenário gerou NaNs.")
        st.stop()

    Xf = future_block[features].values
    pred_f, proba_f = predict_proba_batch(model, scaler, Xf, threshold)

    future_block["P(ALTA)"] = proba_f
    future_block["Sinal"] = np.where(pred_f == 1, "ALTA", "BAIXA")

    # pega a previsão exatamente na data alvo (se existir), senão a última válida antes dela
    alvo_ts = pd.to_datetime(alvo)
    if (future_block["Data"] == alvo_ts).any():
        row = future_block.loc[future_block["Data"] == alvo_ts].iloc[0]
    else:
        row = future_block.iloc[-1]

    sinal_alvo = 1 if float(row["P(ALTA)"]) >= threshold else 0
    proba_alvo = float(row["P(ALTA)"])
    data_real_alvo = pd.to_datetime(row["Data"]).date()

    if sinal_alvo == 1:
        st.success(f"📈 Tendência prevista para **{data_real_alvo}**: **ALTA** — P(ALTA)={proba_alvo:.2%}")
    else:
        st.warning(f"📉 Tendência prevista para **{data_real_alvo}**: **BAIXA** — P(ALTA)={proba_alvo:.2%}")

    st.dataframe(future_block[["Data", "Último", "P(ALTA)", "Sinal"]], use_container_width=True)

    fig2 = make_signal_chart(
        df_plot=future_block,
        pred=pred_f,
        proba=proba_f,
        threshold=threshold,
        title=f"Simulação futura — sinais do modelo (até {alvo})",
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab_diag:
    st.subheader("Painel explícito de métricas (fixas do Colab — sem re-treino)")

    st.caption(f"Modelo: {METRICAS_COLAB['modelo']}")
    st.caption(f"Validação: {METRICAS_COLAB['janela_validacao']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acurácia Treino", f"{METRICAS_COLAB['acc_train']*100:.2f}%")
    c2.metric("Acurácia Teste", f"{METRICAS_COLAB['acc_test']*100:.2f}%")
    c3.metric("Overfitting", f"{METRICAS_COLAB['overfit']*100:.2f}%")
    c4.metric("F1 (CV)", f"{METRICAS_COLAB['cv_f1_mean']:.3f} ± {METRICAS_COLAB['cv_f1_pm']:.3f}")

    st.divider()

    cm = METRICAS_COLAB["cm"]
    cm_df = pd.DataFrame(
        cm,
        index=["Real: Queda (0)", "Real: Alta (1)"],
        columns=["Prev: Queda (0)", "Prev: Alta (1)"]
    )

    colA, colB = st.columns([1, 1.2])
    with colA:
        st.write("Matriz de confusão (tabela):")
        st.dataframe(cm_df, use_container_width=True)

    with colB:
        st.write("Matriz de confusão (gráfico):")
        st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

    st.divider()

    st.write("Classification report (do Colab):")
    st.code(METRICAS_COLAB["report"])

    st.divider()

    st.write("Diagnóstico rápido do dataset carregado (para auditoria):")
    d1, d2, d3 = st.columns(3)
    d1.metric("Linhas válidas (features)", len(df))
    d2.metric("Data inicial", str(df["Data"].iloc[0].date()))
    d3.metric("Data final", str(df["Data"].iloc[-1].date()))

    st.write("Resumo do `Último` (corrigido):")
    st.write(df["Último"].describe())

    st.write("Últimos 10 pontos (Data, Último):")
    st.dataframe(df[["Data", "Último"]].tail(10), use_container_width=True)
