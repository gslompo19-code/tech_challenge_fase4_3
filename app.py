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
# Utilitários: modelo/scaler e log
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

    # fallback: treina se não existir (pra não quebrar app)
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
        "timestamp", "source", "rows", "last_date",
        "pred_direction", "pred_proba", "test_n", "acc_test", "f1_test"
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


def append_log(source, df, pred_direction, pred_proba, test_n, acc, f1):
    ensure_log()
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "rows": int(len(df)),
        "last_date": str(df["Data"].iloc[-1]),
        "pred_direction": int(pred_direction),
        "pred_proba": float(pred_proba) if pred_proba is not None else np.nan,
        "test_n": int(test_n),
        "acc_test": float(acc),
        "f1_test": float(f1),
    }])
    row.to_csv(LOG_PATH, mode="a", header=False, index=False)


def predict_next_signal(model, scaler, X_last_raw):
    X_last = scaler.transform(X_last_raw)
    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(X_last)[0, 1])
        yhat = int(p >= 0.5)
        return yhat, p
    # fallback
    yhat = int(model.predict(X_last)[0])
    return yhat, None


# =========================
# UI: Sidebar
# =========================
st.title("📈 IBOV Signal — Sistema Preditivo (Fase 4)")
st.caption("Foco em produto: gerar um sinal preditivo para o próximo dia, com monitoramento em abas separadas.")

with st.sidebar:
    st.header("Fonte de dados")
    uploaded = st.file_uploader("Upload de CSV (opcional)", type=["csv"])
    test_n = st.number_input("Janela de teste (últimos N)", min_value=10, max_value=200, value=30, step=5)

    st.header("Decisão (produto)")
    threshold = st.slider("Threshold de decisão (P[ALTA] ≥ t)", 0.30, 0.70, 0.50, 0.01)

    st.header("Visibilidade")
    show_logs = st.checkbox("Mostrar logs", value=False)


# =========================
# Carregar CSV (repo por padrão)
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
        st.error(
            f"Não encontrei '{DEFAULT_CSV}' no repositório.\n"
            "Renomeie o arquivo para algo simples (ex.: dados_ibovespa.csv) e atualize DEFAULT_CSV."
        )
        st.stop()

try:
    df = carregar_dados(csv_path)
except Exception as e:
    st.error(f"Erro ao carregar/processar CSV: {e}")
    st.stop()

features = df.attrs.get("features_sugeridas", [])
X_raw = df[features].values
y_raw = df["Alvo"].values

# split temporal: últimos N como teste (igual sua lógica)
split_idx = len(X_raw) - int(test_n)
if split_idx <= 0:
    st.error("Dataset pequeno demais para o tamanho de teste escolhido.")
    st.stop()

X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
y_train, y_test = y_raw[:split_idx], y_raw[split_idx:]

# scaler e modelo
scaler = load_scaler_or_fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

model = load_model_or_train(X_train, y_train)

# avaliação básica (monitoramento)
y_pred_test = model.predict(X_test)
acc = accuracy_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

# produto: previsão do "próximo dia" = usando a última linha disponível
X_last_raw = df[features].iloc[[-1]].values
pred_dir, pred_proba = predict_next_signal(model, scaler, X_last_raw)
if pred_proba is not None:
    pred_dir = int(pred_proba >= threshold)

append_log(source_name, df, pred_dir, pred_proba, test_n, acc, f1)

# =========================
# Abas (Produto / Monitoramento / Sobre)
# =========================
tab_produto, tab_monitor, tab_sobre = st.tabs(["🧠 Produto", "📊 Monitoramento", "📘 Sobre o modelo"])

# --------
# PRODUTO
# --------
with tab_produto:
    c1, c2, c3 = st.columns([1.7, 1.0, 1.0])

    with c1:
        st.subheader("Preço (Último)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Data"], y=df["Último"], mode="lines", name="Último"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Sinal para o próximo dia")
        last_date = df["Data"].iloc[-1]
        last_price = float(df["Último"].iloc[-1])
        st.write(f"**Última data no dataset:** {last_date.date()}")
        st.metric("Último preço", f"{last_price:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

        if pred_dir == 1:
            st.success("📈 **Sinal: ALTA** (comprado)")
            st.caption("Interpretação: o modelo estima maior chance de fechamento acima do atual no próximo dia.")
        else:
            st.warning("📉 **Sinal: BAIXA** (fora / vendido)")
            st.caption("Interpretação: o modelo estima menor chance de alta no próximo dia.")

        if pred_proba is not None:
            st.metric("Probabilidade de ALTA", f"{pred_proba:.2%}")
            st.caption(f"Threshold atual: {threshold:.2f}")

    with c3:
        st.subheader("Como usar (regra simples)")
        st.write(
            "- **Entrada:** sinal ALTA.\n"
            "- **Saída:** no fechamento do dia seguinte.\n"
            "- **Objetivo do modelo:** prever se **Último(t+1) > Último(t)**.\n"
            "- **Não é recomendação financeira**: é um protótipo acadêmico."
        )
        st.write(f"Fonte de dados: `{source_name}`")

    st.divider()

    st.subheader("Resumo de decisão do dia")
    resumo = {
        "Data base": str(df["Data"].iloc[-1].date()),
        "Preço base": float(df["Último"].iloc[-1]),
        "Sinal": "ALTA" if pred_dir == 1 else "BAIXA",
        "Prob(ALTA)": (float(pred_proba) if pred_proba is not None else None),
        "Threshold": float(threshold),
    }
    st.json(resumo)

# ----------------
# MONITORAMENTO
# ----------------
with tab_monitor:
    st.subheader("Métricas do modelo (janela de teste)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia (teste)", f"{acc:.2%}")
    c2.metric("F1 (teste)", f"{f1:.3f}")
    c3.metric("Janela de teste", f"{int(test_n)} últimos registros")

    if acc >= 0.75:
        st.success("✅ Acurácia ≥ 75% na janela de teste configurada.")
    else:
        st.warning("⚠️ Acurácia < 75%. Ajuste N do teste ou valide se seus .pkl correspondem ao treino do notebook.")

    st.divider()

    st.subheader("Matriz de confusão / relatório")
    st.write(confusion_matrix(y_test, y_pred_test))
    st.text(classification_report(y_test, y_pred_test))

    st.divider()

    st.subheader("Real vs Previsto (teste)")
    tabela = pd.DataFrame({
        "Data": df["Data"].iloc[-int(test_n):].values,
        "Real": y_test,
        "Previsto": y_pred_test
    })
    tabela["Acerto"] = np.where(tabela["Real"] == tabela["Previsto"], "✔️", "❌")
    st.dataframe(tabela, use_container_width=True)

    if show_logs:
        st.divider()
        st.subheader("Logs (predições realizadas no app)")
        ensure_log()
        log_df = pd.read_csv(LOG_PATH, on_bad_lines="skip")
        st.dataframe(log_df.tail(50), use_container_width=True)

# ----------
# SOBRE
# ----------
with tab_sobre:
    st.subheader("O que este sistema faz")
    st.write(
        "Este app é um **sistema preditivo** que gera um sinal (ALTA/BAIXA) para o próximo dia "
        "a partir de indicadores técnicos calculados sobre o histórico do IBOV."
    )

    st.subheader("Definição do alvo")
    st.code('Alvo = 1 se Último(t+1) > Último(t), senão 0', language="text")

    st.subheader("Prevenção de vazamento")
    st.write(
        "- Features são calculadas com janelas passadas.\n"
        "- Split temporal: treino antes, teste nos últimos N.\n"
        "- `MinMaxScaler` com `fit` apenas no treino."
    )

    st.subheader("Features usadas")
    st.write(features)

    st.subheader("Arquivos do repositório recomendados")
    st.write(
        "- `MODEL_CARD.md` (estratégia, hipóteses, limitações)\n"
        "- `README.md` (como rodar + link do app)\n"
        "- `notebook.ipynb` (experimentos)\n"
        "- `modelo_catboost.pkl` e `scaler_minmax.pkl` (artefatos)"
    )
