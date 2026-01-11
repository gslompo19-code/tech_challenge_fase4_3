import os
from datetime import datetime, timedelta

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
# Funções do Colab (iguais/compatíveis)
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


def carregar_dados(caminho_csv):
    """
    Versão baseada no seu Colab.
    IMPORTANTE: a inferência do Streamlit deve gerar as MESMAS features do treino do modelo.
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
# Utilitários do app
# =========================
@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Não encontrei {MODEL_PATH} no repositório.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Não encontrei {SCALER_PATH} no repositório.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_features_df(csv_path):
    df = carregar_dados(csv_path)
    features = df.attrs["features_sugeridas"]
    return df, features


def predict_row(model, scaler, df, features, idx, threshold=0.50):
    X = df.loc[[idx], features].values
    Xs = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(Xs)[0, 1])
    else:
        p = float(model.predict(Xs)[0])
    y = int(p >= threshold)
    return y, p


def make_signal_chart(df_plot, price_col, date_col, pred, proba, threshold, title):
    price_vals = df_plot[price_col].astype(float).values
    dates = df_plot[date_col]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=("Preço + Sinal (ALTA/BAIXA)", "Probabilidade P(ALTA)")
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
    fig.add_hline(
        y=threshold, line_dash="dash", line_width=2,
        annotation_text=f"threshold={threshold:.2f}", row=2, col=1
    )

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


# =========================
# App
# =========================
st.title("📈 IBOV Signal — Sistema Preditivo (somente inferência do modelo do Colab)")

with st.sidebar:
    st.header("Arquivos (do repositório)")
    st.write(f"- CSV: `{DEFAULT_CSV}`")
    st.write(f"- Modelo: `{MODEL_PATH}`")
    st.write(f"- Scaler: `{SCALER_PATH}`")

    st.header("Config do produto")
    threshold = st.slider("Threshold para decidir ALTA", 0.30, 0.70, 0.50, 0.01)
    view_n = st.slider("Janela do gráfico (últimos N)", 60, 1500, 400, 20)

    st.caption("Obs: Acurácia do Colab (80%) é do backtest (últimos 30). Aqui é um sistema preditivo/interativo.")


# valida presença do CSV
if not os.path.exists(DEFAULT_CSV):
    st.error(
        f"Não encontrei o arquivo `{DEFAULT_CSV}` no repositório. "
        "Suba esse CSV junto do app.py para o Streamlit não pedir upload."
    )
    st.stop()

# load
try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(str(e))
    st.stop()

df, features = load_features_df(DEFAULT_CSV)

# =========================
# Tabs
# =========================
tab_produto, tab_futuro, tab_sobre = st.tabs(["🧠 Produto", "🔮 Simulação futura (30 dias)", "📄 Sobre o modelo"])

with tab_produto:
    st.subheader("Produto: selecione uma data do histórico e receba a tendência do dia seguinte")

    date_options = df["Data"].dt.date.tolist()
    selected_date = st.selectbox("Escolha uma data (histórico)", options=date_options, index=len(date_options) - 1)

    idx_list = df.index[df["Data"].dt.date == selected_date]
    idx = int(idx_list[0])

    y, p = predict_row(model, scaler, df, features, idx, threshold=threshold)
    if y == 1:
        st.success(f"📈 Tendência prevista (dia seguinte): **ALTA** — P(ALTA)={p:.2%}")
    else:
        st.warning(f"📉 Tendência prevista (dia seguinte): **BAIXA** — P(ALTA)={p:.2%}")

    st.write("Dados do dia selecionado:")
    st.dataframe(df.loc[[idx], ["Data", "Último", "Vol.", "rsi", "macd", "bb_largura", "atr_pct", "Alvo"]], use_container_width=True)

    # gráfico produto (histórico + sinal)
    df_plot = df.tail(int(view_n)).copy()
    X_all = df_plot[features].values
    Xs_all = scaler.transform(X_all)

    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(Xs_all)[:, 1]
    else:
        proba_all = model.predict(Xs_all).astype(float)

    pred_all = (proba_all >= threshold).astype(int)

    fig = make_signal_chart(
        df_plot=df_plot,
        price_col="Último",
        date_col="Data",
        pred=pred_all,
        proba=proba_all,
        threshold=threshold,
        title="Histórico + Sinais do modelo (interativo)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Nota: o campo `Alvo` é o que aconteceu no dado histórico. O sistema mostra o que o modelo sinaliza para o próximo dia a partir das features do dia escolhido.")


with tab_futuro:
    st.subheader("Simulação futura (cenários) por 30 dias")
    st.write(
        "Como não existe preço real futuro no dataset, esta tela faz **cenários**. "
        "Você escolhe uma regra de retorno (ex.: +0,2% ao dia), geramos uma série de preços futura e "
        "rodamos o **modelo do Colab** para classificar ALTA/BAIXA dia a dia."
    )

    last_date = df["Data"].iloc[-1]
    last_price = float(df["Último"].iloc[-1])
    st.info(f"Último ponto do dataset: {last_date.date()} — Último={last_price:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    mode = st.selectbox("Cenário de preços (retornos diários)", ["Constante", "Constante + Ruído", "Aleatório (volatilidade)"])

    if mode == "Constante":
        mu = st.number_input("Retorno diário (%)", value=0.20, step=0.05) / 100.0
        sigma = 0.0
    elif mode == "Constante + Ruído":
        mu = st.number_input("Retorno médio diário (%)", value=0.15, step=0.05) / 100.0
        sigma = st.number_input("Ruído diário (%)", value=0.30, step=0.05) / 100.0
    else:
        mu = st.number_input("Retorno médio diário (%)", value=0.05, step=0.05) / 100.0
        sigma = st.number_input("Volatilidade diária (%)", value=0.80, step=0.05) / 100.0

    seed = st.number_input("Seed (reprodutibilidade)", value=42, step=1)
    np.random.seed(int(seed))

    horizon = 30
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]

    rets = np.random.normal(loc=mu, scale=sigma, size=horizon)
    prices = [last_price]
    for r in rets:
        prices.append(prices[-1] * (1.0 + r))
    future_prices = prices[1:]

    # monta DF futuro “mínimo” para recalcular features:
    # - para não inventar OHLC e volume, usamos aproximações simples:
    #   Abertura/Máxima/Mínima ~ Último com pequenas variações
    #   Vol. mantém o último volume conhecido
    base = df.copy()
    last_vol = float(base["Vol."].iloc[-1]) if pd.notna(base["Vol."].iloc[-1]) else 0.0

    fut = pd.DataFrame({
        "Data": future_dates,
        "Último": future_prices,
        "Abertura": future_prices,
        "Máxima": [p * 1.002 for p in future_prices],
        "Mínima": [p * 0.998 for p in future_prices],
        "Vol.": [last_vol for _ in range(horizon)],
    })

    # concat e recalcula features pelo MESMO pipeline
    # (mantendo colunas esperadas pela carregar_dados)
    # Para isso, criamos um CSV-like DF com as mesmas colunas originais do seu arquivo:
    # Como o carregar_dados lê CSV, aqui recalculamos “na sequência” copiando a lógica
    # mas sem reler arquivo (para não depender de formatação string).
    full = pd.concat([base[["Data","Vol.","Último","Abertura","Máxima","Mínima"]], fut], ignore_index=True)
    full = full.sort_values("Data").reset_index(drop=True)

    # Recalcular features (mesma lógica do carregar_dados a partir do ponto 3)
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

    # OBV requer “variação do Último” e Vol.
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

    # pega somente linhas futuras com features completas
    future_block = full[full["Data"].isin(future_dates)].copy()
    future_block = future_block.dropna(subset=features)

    if len(future_block) == 0:
        st.error("Não foi possível calcular features suficientes para o horizonte. Aumente histórico/ajuste cenário.")
        st.stop()

    Xf = future_block[features].values
    Xf_s = scaler.transform(Xf)

    if hasattr(model, "predict_proba"):
        proba_f = model.predict_proba(Xf_s)[:, 1]
    else:
        proba_f = model.predict(Xf_s).astype(float)

    pred_f = (proba_f >= threshold).astype(int)
    future_block["P(ALTA)"] = proba_f
    future_block["Sinal"] = np.where(pred_f == 1, "ALTA", "BAIXA")

    st.write("Resultado da simulação (30 dias):")
    st.dataframe(
        future_block[["Data", "Último", "P(ALTA)", "Sinal"]].tail(30),
        use_container_width=True
    )

    # gráfico futuro
    fig2 = make_signal_chart(
        df_plot=future_block.tail(30),
        price_col="Último",
        date_col="Data",
        pred=pred_f[-30:],
        proba=proba_f[-30:],
        threshold=threshold,
        title="Simulação futura (cenário) — sinais do modelo",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Importante: aqui você está simulando um cenário de preços e o modelo classifica tendências com base nesse cenário. "
        "Isso atende o requisito de 'testar o modelo e prever novas tendências' sem re-treinar."
    )


with tab_sobre:
    st.subheader("Estratégia do modelo (resumo)")
    st.write(
        "- Modelo: CatBoostClassifier treinado no Colab\n"
        "- Features: retornos, volatilidades, RSI, MACD, Bollinger, ATR, OBV e z-scores\n"
        "- Alvo (Alvo): 1 se o preço do dia seguinte sobe, 0 caso contrário\n"
        "- Este app NÃO re-treina; apenas carrega os .pkl e faz inferência.\n"
    )

    st.write("Features usadas:")
    st.code("\n".join(features))

    st.write("Diagnóstico rápido do dataset carregado:")
    c1, c2, c3 = st.columns(3)
    c1.metric("Linhas (após dropna features)", len(df))
    c2.metric("Data inicial", str(df["Data"].iloc[0].date()))
    c3.metric("Data final", str(df["Data"].iloc[-1].date()))
