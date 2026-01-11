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
        "timestamp", "source", "mode",
        "selected_date", "selected_price",
        "pred_direction", "pred_proba",
        "threshold", "test_n", "acc_test", "f1_test"
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


def append_log(source, mode, selected_date, selected_price, pred_direction, pred_proba, threshold, test_n, acc, f1):
    ensure_log()
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "mode": mode,
        "selected_date": selected_date,
        "selected_price": selected_price,
        "pred_direction": int(pred_direction),
        "pred_proba": float(pred_proba) if pred_proba is not None else np.nan,
        "threshold": float(threshold),
        "test_n": int(test_n),
        "acc_test": float(acc),
        "f1_test": float(f1),
    }])
    row.to_csv(LOG_PATH, mode="a", header=False, index=False)


def predict_direction(model, X_scaled, threshold=0.5):
    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(X_scaled)[0, 1])
        yhat = int(p >= threshold)
        return yhat, p
    yhat = int(model.predict(X_scaled)[0])
    return yhat, None


# =========================
# UI
# =========================
st.title("📈 IBOV Signal — Interativo (Histórico + Simulação)")
st.caption("Modo histórico (recomendado): selecione uma data e receba a tendência para o dia seguinte.")

with st.sidebar:
    st.header("Fonte de dados")
    uploaded = st.file_uploader("Upload de CSV (opcional)", type=["csv"])
    test_n = st.number_input("Janela de teste (últimos N)", min_value=10, max_value=200, value=30, step=5)

    st.header("Decisão")
    threshold = st.slider("Threshold P(ALTA) ≥ t", 0.30, 0.70, 0.50, 0.01)

    st.header("Modo interativo")
    mode = st.radio("Escolha o modo", ["Histórico (selecionar data)", "Manual (simulação)"], index=0)

    st.header("Logs")
    show_logs = st.checkbox("Mostrar logs", value=False)

# CSV
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

# data
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

# monitoramento
y_pred_test = model.predict(X_test)
acc = accuracy_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

# abas
tab_produto, tab_monitor, tab_sobre = st.tabs(["🧠 Produto", "📊 Monitoramento", "📘 Sobre"])

# =========================
# PRODUTO
# =========================
with tab_produto:
    st.subheader("Interação do usuário")

    c1, c2 = st.columns([1.4, 1.0])

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Data"], y=df["Último"], mode="lines", name="Último"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    selected_date = None
    selected_price = None
    pred_dir = None
    pred_proba = None

    with c2:
        if mode == "Histórico (selecionar data)":
            st.write("✅ **Modo recomendado**: usa o histórico real para calcular indicadores e prever o dia seguinte.")
            # só permite selecionar datas que existam no df
            date_options = df["Data"].dt.date.tolist()
            default_date = df["Data"].iloc[-1].date()
            selected_date = st.selectbox("Selecione uma data do histórico", options=date_options, index=len(date_options)-1)

            idx = df.index[df["Data"].dt.date == selected_date]
            if len(idx) == 0:
                st.error("Data não encontrada no histórico.")
            else:
                i = int(idx[0])
                selected_price = float(df.loc[i, "Último"])

                # previsão para o dia seguinte usa features do dia selecionado
                X_sel_raw = df.loc[[i], features].values
                X_sel = scaler.transform(X_sel_raw)

                pred_dir, pred_proba = predict_direction(model, X_sel, threshold)

                # label
                if pred_dir == 1:
                    st.success("📈 Tendência prevista: **ALTA** (dia seguinte)")
                else:
                    st.warning("📉 Tendência prevista: **BAIXA** (dia seguinte)")

                st.metric("Preço na data selecionada", f"{selected_price:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

                if pred_proba is not None:
                    st.metric("Probabilidade de ALTA", f"{pred_proba:.2%}")
                    st.caption(f"Threshold: {threshold:.2f}")

                # log
                append_log(
                    source=source_name,
                    mode="historico",
                    selected_date=str(selected_date),
                    selected_price=selected_price,
                    pred_direction=pred_dir,
                    pred_proba=pred_proba,
                    threshold=threshold,
                    test_n=test_n,
                    acc=acc,
                    f1=f1
                )

        else:
            st.write("⚠️ **Modo simulação**: você altera o preço manualmente. Isso é uma aproximação.")
            st.caption("Como o modelo usa indicadores (RSI/MACD/Bollinger), o jeito correto é selecionar uma data do histórico.")

            # escolhe uma data base para pegar os indicadores reais e simular só o preço do dia
            base_date = df["Data"].iloc[-1].date()
            base_date = st.selectbox("Data base (para indicadores)", options=df["Data"].dt.date.tolist(), index=len(df)-1)

            idx = df.index[df["Data"].dt.date == base_date]
            i = int(idx[0])

            base_price = float(df.loc[i, "Último"])
            selected_price = st.number_input("Digite um preço (Último) para simular", min_value=0.0, value=base_price, step=10.0)

            # Copia a linha de features e faz uma simulação mínima:
            # - recalcula apenas ret_1d/log_ret/ret_5d e z_close_20 a partir do preço simulado (aproximado)
            # - mantém os demais indicadores iguais ao do dia base (limitação assumida)
            row = df.loc[i].copy()
            prev_close = float(df.loc[i-1, "Último"]) if i > 0 else base_price

            # atualiza colunas mínimas dependentes do preço
            row["ret_1d"] = (selected_price / prev_close) - 1 if prev_close != 0 else 0.0
            row["log_ret"] = np.log(selected_price) - np.log(prev_close) if prev_close > 0 and selected_price > 0 else 0.0

            # ret_5d aproximado
            if i >= 5:
                close_5 = float(df.loc[i-5, "Último"])
                row["ret_5d"] = (selected_price / close_5) - 1 if close_5 != 0 else 0.0
            else:
                row["ret_5d"] = np.nan

            # z_close_20 aproximado usando média/std histórica até i (sem recalcular todo)
            if i >= 20:
                hist = df.loc[i-19:i, "Último"].astype(float).copy()
                hist.iloc[-1] = selected_price
                m = hist.mean()
                sd = hist.std()
                row["z_close_20"] = (selected_price - m) / sd if sd and not np.isnan(sd) else 0.0
            else:
                row["z_close_20"] = np.nan

            # prepara X
            X_sel_raw = pd.DataFrame([row])[features].values
            # se houver NaN por falta de janela, impede predição
            if np.isnan(X_sel_raw).any():
                st.error("Não dá pra simular essa data: faltam janelas (ex.: 20 dias) para algumas features.")
            else:
                X_sel = scaler.transform(X_sel_raw)
                pred_dir, pred_proba = predict_direction(model, X_sel, threshold)

                if pred_dir == 1:
                    st.success("📈 Tendência prevista (simulada): **ALTA**")
                else:
                    st.warning("📉 Tendência prevista (simulada): **BAIXA**")

                if pred_proba is not None:
                    st.metric("Probabilidade de ALTA", f"{pred_proba:.2%}")

                append_log(
                    source=source_name,
                    mode="manual_simulacao",
                    selected_date=str(base_date),
                    selected_price=float(selected_price),
                    pred_direction=pred_dir,
                    pred_proba=pred_proba,
                    threshold=threshold,
                    test_n=test_n,
                    acc=acc,
                    f1=f1
                )

    st.divider()
    st.subheader("Resumo")
    st.write(
        "🧠 **Produto**: o usuário escolhe uma data (ou simula preço) e recebe um **sinal** para o dia seguinte.\n"
        "📌 **Recomendado**: modo histórico."
    )

# =========================
# MONITORAMENTO
# =========================
with tab_monitor:
    st.subheader("Saúde do modelo (monitoramento)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia (teste)", f"{acc:.2%}")
    c2.metric("F1 (teste)", f"{f1:.3f}")
    c3.metric("Teste (últimos N)", f"{int(test_n)}")

    if acc >= 0.75:
        st.success("✅ Acurácia ≥ 75% na janela de teste configurada.")
    else:
        st.warning("⚠️ Acurácia < 75%. Tente N maior (ex.: 60/90) para uma medida mais estável.")

    st.divider()
    st.subheader("Matriz de confusão / relatório")
    st.write(confusion_matrix(y_test, y_pred_test))
    st.text(classification_report(y_test, y_pred_test))

    if show_logs:
        st.divider()
        st.subheader("Logs")
        ensure_log()
        log_df = pd.read_csv(LOG_PATH, on_bad_lines="skip")
        st.dataframe(log_df.tail(50), use_container_width=True)
def simular_futuro_cenario(df_base: pd.DataFrame, dias_a_frente: int, retorno_diario: float):
    """
    Simula um caminho de preços para datas futuras e recalcula as features.
    retorno_diario em decimal (ex.: 0.002 = +0,2% ao dia)
    """
    df_sim = df_base.copy()

    last_date = df_sim["Data"].iloc[-1]
    last_close = float(df_sim["Último"].iloc[-1])

    # cria datas futuras (dias corridos)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=dias_a_frente, freq="D")

    # cria preços futuros por retorno constante
    future_close = []
    price = last_close
    for _ in range(dias_a_frente):
        price = price * (1 + retorno_diario)
        future_close.append(price)

    # adiciona linhas futuras com colunas mínimas
    future_df = pd.DataFrame({
        "Data": future_dates,
        "Último": future_close,
        # aprox: mantém Abertura/Máxima/Mínima iguais ao Último (simples)
        "Abertura": future_close,
        "Máxima": future_close,
        "Mínima": future_close,
        # volume: mantém último volume conhecido (simples)
        "Vol.": [float(df_sim["Vol."].iloc[-1])] * dias_a_frente,
    })

    # concatena e recalcula TODAS as features com as mesmas regras do seu pipeline
    df_all = pd.concat([df_sim, future_df], ignore_index=True)

    # Recalcular features (repetindo a lógica do carregar_dados, mas agora df já é numérico)
    df_all = df_all.sort_values("Data").reset_index(drop=True)

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
    df_all["bb_sup"] = bb_media + 2*bb_std
    df_all["bb_inf"] = bb_media - 2*bb_std
    df_all["bb_largura"] = (df_all["bb_sup"] - df_all["bb_inf"]) / bb_media

    tr1 = df_all["Máxima"] - df_all["Mínima"]
    tr2 = (df_all["Máxima"] - df_all["Último"].shift(1)).abs()
    tr3 = (df_all["Mínima"] - df_all["Último"].shift(1)).abs()
    df_all["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_all["ATR"] = df_all["TR"].rolling(14, min_periods=14).mean()

    # OBV recalculado (usa Vol. e direção do preço)
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

    features = df_base.attrs.get("features_sugeridas")
    if not features:
        features = [
            "ret_1d","log_ret","ret_5d","rv_20",
            "atr_pct","bb_largura","desvio_mm3_pct",
            "vol_log","vol_ret","obv_diff",
            "rsi","macd","sinal_macd","hist_macd",
            "dia","z_close_20","z_rsi_20","z_macd_20"
        ]

    # pega somente as linhas futuras
    df_future = df_all.tail(dias_a_frente).copy()
    return df_future, features
df_future, features = simular_futuro_cenario(df, dias_a_frente=10, retorno_diario=0.002)

X_future_raw = df_future[features].values
X_future = scaler.transform(X_future_raw)

proba = model.predict_proba(X_future)[:, 1]
pred = (proba >= threshold).astype(int)

out = pd.DataFrame({
    "Data": df_future["Data"].dt.date,
    "Preço Simulado": df_future["Último"].astype(float),
    "Prob(ALTA)": proba,
    "Sinal": np.where(pred == 1, "ALTA", "BAIXA")
})

# =========================
# SOBRE
# =========================
with tab_sobre:
    st.subheader("O que o modelo prevê")
    st.write("**Alvo:** 1 se `Último(t+1) > Último(t)` senão 0.")

    st.subheader("Como prevenimos vazamento")
    st.write("- Split temporal (treino antes, teste nos últimos N)\n- Scaler com fit apenas no treino")

    st.subheader("Features usadas")
    st.write(features)

    st.subheader("Sugestão para o GitHub (documento à parte)")
    st.write("Crie um `MODEL_CARD.md` explicando objetivo, dados, estratégia, validação e limitações.")


