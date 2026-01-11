import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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


def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv)
    df.columns = df.columns.str.strip()

    df["Data"] = pd.to_datetime(df["Data"], format="%d.%m.%Y", errors="coerce")
    if df["Data"].isna().mean() > 0.5:
        df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
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
        "threshold", "window_backtest"
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


def append_log(source, action, selected_date, pred_direction, pred_proba, threshold, window_backtest):
    ensure_log()
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "action": action,
        "selected_date": selected_date,
        "pred_direction": int(pred_direction) if pred_direction is not None else np.nan,
        "pred_proba": float(pred_proba) if pred_proba is not None else np.nan,
        "threshold": float(threshold),
        "window_backtest": int(window_backtest) if window_backtest is not None else np.nan,
    }])
    row.to_csv(LOG_PATH, mode="a", header=False, index=False)


def predict_proba_or_none(model, X_scaled):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_scaled)[:, 1]
    return None


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

    # recalcula features (mesma lógica)
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
st.title("📈 IBOV Signal — Gráficos Interativos + Análises Temporais")
st.caption("Selecione uma data para prever o dia seguinte, faça backtest em janelas e simule cenários futuros.")

with st.sidebar:
    st.header("Dados")
    uploaded = st.file_uploader("Upload de CSV (opcional)", type=["csv"])
    test_n = st.number_input("Janela de teste (últimos N)", min_value=10, max_value=260, value=60, step=10)

    st.header("Decisão")
    threshold = st.slider("Threshold P(ALTA) ≥ t", 0.30, 0.70, 0.50, 0.01)

    st.header("Backtest (visual)")
    backtest_window = st.number_input("Backtest: últimos N dias", min_value=60, max_value=600, value=250, step=10)

    st.header("Rolling métricas")
    rolling_window = st.number_input("Janela rolling (dias)", min_value=20, max_value=200, value=60, step=10)

    st.header("Simulação futura")
    fut_days = st.number_input("Dias à frente", min_value=1, max_value=30, value=10, step=1)
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

scaler = load_scaler_or_fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

model = load_model_or_train(X_train, y_train)

# =========================
# Probabilidades e sinais (para gráficos)
# =========================
X_all_scaled = scaler.transform(X_raw)
proba_all = predict_proba_or_none(model, X_all_scaled)
if proba_all is None:
    proba_all = model.predict(X_all_scaled).astype(float)

pred_all = (proba_all >= threshold).astype(int)

# =========================
# Tabs
# =========================
tab_prod, tab_backtest, tab_future, tab_about = st.tabs(
    ["🧠 Produto (interativo)", "📊 Backtest & Temporal", "🔮 Simulação Futura", "📘 Sobre"]
)

# ======================================================
# TAB 1: PRODUTO INTERATIVO (selecionar data e prever t+1)
# ======================================================
with tab_prod:
    st.subheader("Selecione uma data e veja a tendência para o dia seguinte")

    # datas disponíveis
    date_options = df["Data"].dt.date.tolist()
    default_date = df["Data"].iloc[-1].date()
    selected_date = st.selectbox("Data (histórico)", options=date_options, index=len(date_options) - 1)

    idx_list = df.index[df["Data"].dt.date == selected_date]
    if len(idx_list) == 0:
        st.error("Data não encontrada.")
        st.stop()

    i = int(idx_list[0])

    # previsão "para o dia seguinte": usa features do dia selecionado
    X_sel_raw = df.loc[[i], features].values
    X_sel = scaler.transform(X_sel_raw)
    p_sel = float(predict_proba_or_none(model, X_sel)[0]) if hasattr(model, "predict_proba") else float(model.predict(X_sel)[0])
    y_sel = int(p_sel >= threshold)

    if y_sel == 1:
        st.success(f"📈 Tendência prevista para o dia seguinte: **ALTA** (P(ALTA)={p_sel:.2%})")
    else:
        st.warning(f"📉 Tendência prevista para o dia seguinte: **BAIXA** (P(ALTA)={p_sel:.2%})")

    append_log(source_name, "predict_by_date", str(selected_date), y_sel, p_sel, threshold, backtest_window)

    # gráfico interativo: preço + prob + marcador na data selecionada
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Data"], y=df["Último"], mode="lines", name="Preço (Último)"
    ))

    fig.add_trace(go.Scatter(
        x=df["Data"], y=proba_all, mode="lines", name="Probabilidade de ALTA",
        yaxis="y2"
    ))

    # marcadores de sinal
    fig.add_trace(go.Scatter(
        x=df["Data"], y=df["Último"],
        mode="markers",
        name="Sinal (ALTA=1)",
        marker=dict(size=6),
        text=[f"Sinal={int(s)} | P(ALTA)={float(p):.2%}" for s, p in zip(pred_all, proba_all)],
        hovertemplate="%{x}<br>%{text}<br>Preço=%{y}<extra></extra>"
    ))

    # destaque da data selecionada
    fig.add_vline(x=pd.to_datetime(selected_date), line_width=2)

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Preço"),
        yaxis2=dict(title="P(ALTA)", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # mini resumo temporal do ponto selecionado
    c1, c2, c3 = st.columns(3)
    c1.metric("Preço na data", f"{float(df.loc[i,'Último']):,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    c2.metric("P(ALTA)", f"{p_sel:.2%}")
    c3.metric("Sinal", "ALTA" if y_sel == 1 else "BAIXA")

# ======================================================
# TAB 2: BACKTEST & ANÁLISES TEMPORAIS
# ======================================================
with tab_backtest:
    st.subheader("Backtest visual (estratégia do sinal vs Buy & Hold)")

    # janela do backtest no final do histórico
    bt_n = int(min(backtest_window, len(df) - 2))
    bt_df = df.iloc[-bt_n:].copy()

    # alinhar previsões ao bt_df
    start_idx = len(df) - bt_n
    bt_proba = proba_all[start_idx:start_idx + bt_n]
    bt_pred = pred_all[start_idx:start_idx + bt_n]
    bt_real = bt_df["Alvo"].values.astype(int)

    # retorno do próximo dia (t->t+1) baseado no preço
    close = bt_df["Último"].astype(float).values
    next_ret = np.zeros_like(close, dtype=float)
    next_ret[:-1] = (close[1:] / close[:-1]) - 1
    next_ret[-1] = 0.0

    # estratégia: entra quando sinal=ALTA e sai no dia seguinte (retorno do próximo dia)
    strat_ret = next_ret * (bt_pred.astype(float))
    buyhold_ret = next_ret

    bt_df["P(ALTA)"] = bt_proba
    bt_df["Sinal"] = bt_pred
    bt_df["Ret_Prox_Dia"] = next_ret
    bt_df["Ret_Estrategia"] = strat_ret
    bt_df["Acertou"] = (bt_pred == bt_real).astype(int)

    # acumulados
    bt_df["BH_Acumulado"] = (1 + pd.Series(buyhold_ret)).cumprod()
    bt_df["Estrat_Acumulado"] = (1 + pd.Series(strat_ret)).cumprod()

    c1, c2, c3 = st.columns(3)
    c1.metric("Retorno acumulado (Buy&Hold)", f"{(bt_df['BH_Acumulado'].iloc[-1]-1):.2%}")
    c2.metric("Retorno acumulado (Estratégia)", f"{(bt_df['Estrat_Acumulado'].iloc[-1]-1):.2%}")
    c3.metric("Taxa de acerto (janela)", f"{bt_df['Acertou'].mean():.2%}")

    # gráfico: acumulado
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=bt_df["Data"], y=bt_df["BH_Acumulado"], mode="lines", name="Buy & Hold (acum.)"))
    fig2.add_trace(go.Scatter(x=bt_df["Data"], y=bt_df["Estrat_Acumulado"], mode="lines", name="Estratégia do Sinal (acum.)"))
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Métricas temporais (rolling)")

    # rolling accuracy/f1 na janela do backtest
    rw = int(min(rolling_window, len(bt_df) - 5))
    roll_acc = bt_df["Acertou"].rolling(rw).mean()
    # rolling f1 (aprox: calcular em janelas com função)
    # para manter leve, calculamos f1 por janela com loop pequeno
    roll_f1 = [np.nan] * len(bt_df)
    for k in range(rw - 1, len(bt_df)):
        y_true_w = bt_real[k - rw + 1:k + 1]
        y_pred_w = bt_pred[k - rw + 1:k + 1]
        roll_f1[k] = f1_score(y_true_w, y_pred_w)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=bt_df["Data"], y=roll_acc, mode="lines", name=f"Acurácia rolling ({rw})"))
    fig3.add_trace(go.Scatter(x=bt_df["Data"], y=roll_f1, mode="lines", name=f"F1 rolling ({rw})"))
    fig3.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Real vs Previsto (comunicação da previsão)")
    table = bt_df[["Data", "Último", "P(ALTA)", "Sinal", "Alvo", "Acertou"]].copy()
    table["Resultado"] = np.where(table["Acertou"] == 1, "✔️", "❌")
    st.dataframe(table.tail(120), use_container_width=True)

    st.subheader("Matriz de confusão / relatório (janela de teste)")
    y_pred_test = model.predict(X_test)
    st.write(confusion_matrix(y_test, y_pred_test))
    st.text(classification_report(y_test, y_pred_test))
    st.write(f"Acurácia teste (últimos {int(test_n)}): {accuracy_score(y_test, y_pred_test):.2%} | F1: {f1_score(y_test, y_pred_test):.3f}")

# ======================================================
# TAB 3: SIMULAÇÃO FUTURA
# ======================================================
with tab_future:
    st.subheader("Simulação futura por cenário (criativo e honesto)")
    st.caption(
        "Como datas futuras não têm histórico real, o app permite simular um caminho de preços "
        "com retorno diário constante e gerar sinais condicionais ao cenário."
    )

    retorno_diario = float(fut_ret) / 100.0
    df_future, fcols = simular_futuro_cenario(df, int(fut_days), retorno_diario)

    X_future_raw = df_future[fcols].values
    # se ainda tiver NaN (por causa de janelas), evita quebrar
    if np.isnan(X_future_raw).any():
        st.warning("Para esse cenário/data, algumas features ficaram sem janela (NaN). Aumente histórico ou diminua dias.")
        st.stop()

    X_future = scaler.transform(X_future_raw)
    future_proba = predict_proba_or_none(model, X_future)
    if future_proba is None:
        future_proba = model.predict(X_future).astype(float)

    future_pred = (future_proba >= threshold).astype(int)

    out = pd.DataFrame({
        "Data": df_future["Data"].dt.date,
        "Preço Simulado": df_future["Último"].astype(float),
        "P(ALTA)": future_proba.astype(float),
        "Sinal": np.where(future_pred == 1, "ALTA", "BAIXA"),
    })

    append_log(source_name, "future_scenario", str(df["Data"].iloc[-1].date()), int(future_pred[-1]), float(future_proba[-1]), threshold, backtest_window)

    # gráfico futuro: preço + prob
    figf = go.Figure()
    figf.add_trace(go.Scatter(x=out["Data"], y=out["Preço Simulado"], mode="lines+markers", name="Preço Simulado"))
    figf.add_trace(go.Scatter(x=out["Data"], y=out["P(ALTA)"], mode="lines+markers", name="P(ALTA)", yaxis="y2"))
    figf.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Preço Simulado"),
        yaxis2=dict(title="P(ALTA)", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(figf, use_container_width=True)
    st.dataframe(out, use_container_width=True)

# ======================================================
# TAB 4: SOBRE
# ======================================================
with tab_about:
    st.subheader("O que o modelo prevê")
    st.write("**Alvo:** 1 se `Último(t+1) > Último(t)`, senão 0.")
    st.write("**Uso no app:** gerar um sinal (ALTA/BAIXA) com probabilidade e comunicar isso com análises temporais.")

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
