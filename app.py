import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

from catboost import CatBoostClassifier
import joblib


# =========================
# Config
# =========================
st.set_page_config(page_title="Tech Challenge Fase 4 — IBOV", layout="wide")

MODEL_PATH = "modelo_catboost.pkl"
SCALER_PATH = "scaler_minmax.pkl"
LOG_PATH = os.path.join("logs", "predictions_log.csv")


# =========================
# FUNÇÕES (iguais ao seu notebook)
# =========================
def volume_to_float(value):
    """Transforma volume com sufixos em número decimal."""
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
    """RSI: Índice de Força Relativa."""
    changes = prices.diff()
    gains = changes.clip(lower=0)
    losses = -changes.clip(upper=0)
    avg_gain = gains.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd_components(prices, short=12, long=26, signal=9):
    """MACD e linha de sinal."""
    short_ema = prices.ewm(span=short, adjust=False).mean()
    long_ema = prices.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def obv_series(data):
    """Calcula OBV com base em variação de preço."""
    obv = [0]
    for i in range(1, len(data)):
        if data["Último"].iat[i] > data["Último"].iat[i-1]:
            obv.append(obv[-1] + data["Vol."].iat[i])
        elif data["Último"].iat[i] < data["Último"].iat[i-1]:
            obv.append(obv[-1] - data["Vol."].iat[i])
        else:
            obv.append(obv[-1])
    return obv


def zscore_roll(s: pd.Series, w: int = 20) -> pd.Series:
    m  = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std()
    return (s - m) / sd


def carregar_dados(caminho_csv):
    """
    Criação de indicadores técnicos (igual ao seu notebook).
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
    df["dia"]        = df["Data"].dt.weekday
    df["rsi"]        = calculate_rsi(df["Último"])
    macd, sinal, hist = macd_components(df["Último"])
    df["macd"], df["sinal_macd"], df["hist_macd"] = macd, sinal, hist

    bb_media = df["Último"].rolling(20, min_periods=20).mean()
    bb_std   = df["Último"].rolling(20, min_periods=20).std()
    df["bb_media"]   = bb_media
    df["bb_std"]     = bb_std
    df["bb_sup"]     = bb_media + 2*bb_std
    df["bb_inf"]     = bb_media - 2*bb_std
    df["bb_largura"] = (df["bb_sup"] - df["bb_inf"]) / bb_media

    tr1 = df["Máxima"] - df["Mínima"]
    tr2 = (df["Máxima"] - df["Último"].shift(1)).abs()
    tr3 = (df["Mínima"] - df["Último"].shift(1)).abs()
    df["TR"]  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(14, min_periods=14).mean()

    df["obv"]  = obv_series(df)
    df["Alvo"] = (df["Último"].shift(-1) > df["Último"]).astype("int8")
    df = df.iloc[:-1].copy()

    df["ret_1d"]   = df["Último"].pct_change()
    df["log_ret"]  = np.log(df["Último"]).diff()
    df["ret_5d"]   = df["Último"].pct_change(5)
    df["rv_20"]    = df["ret_1d"].rolling(20, min_periods=20).std()

    df["atr_pct"]        = df["ATR"] / df["Último"]
    df["desvio_mm3_pct"] = (df["desvio_mm3"] / df["mm_3"]).replace([np.inf, -np.inf], np.nan)

    df["vol_log"] = np.log(df["Vol."].clip(lower=1))
    df["vol_ret"] = df["Vol."].pct_change().replace([np.inf, -np.inf], np.nan)

    df["obv_diff"] = pd.Series(df["obv"]).diff()

    df["z_close_20"] = zscore_roll(df["Último"], 20)
    df["z_rsi_20"]   = zscore_roll(df["rsi"], 20)
    df["z_macd_20"]  = zscore_roll(df["macd"], 20)

    features_sugeridas = [
        "ret_1d","log_ret","ret_5d","rv_20",
        "atr_pct","bb_largura","desvio_mm3_pct",
        "vol_log","vol_ret","obv_diff",
        "rsi","macd","sinal_macd","hist_macd",
        "dia","z_close_20","z_rsi_20","z_macd_20"
    ]
    df = df.dropna(subset=features_sugeridas + ["Alvo"]).copy()
    df.attrs["features_sugeridas"] = features_sugeridas
    return df


# =========================
# Log simples de uso
# =========================
def ensure_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=[
            "timestamp", "arquivo", "rows", "last_date", "pred_last",
            "acc_test_30", "f1_test_30"
        ]).to_csv(LOG_PATH, index=False)

def append_log(filename, df, pred_last, acc, f1):
    ensure_log()
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "arquivo": filename,
        "rows": int(len(df)),
        "last_date": str(df["Data"].iloc[-1]),
        "pred_last": int(pred_last),
        "acc_test_30": float(acc),
        "f1_test_30": float(f1),
    }])
    row.to_csv(LOG_PATH, mode="a", header=False, index=False)


# =========================
# UI
# =========================
st.title("📈 Tech Challenge Fase 4 — IBOV (Streamlit)")
st.caption("App baseado diretamente no seu notebook (mesmo pipeline de features + split dos últimos 30 dias).")

with st.sidebar:
    st.header("Entrada")
    uploaded = st.file_uploader("Envie o CSV do IBOV", type=["csv"])

    st.header("Configuração (igual notebook)")
    test_size_last_n = st.number_input("Qtd. últimos registros para teste", min_value=10, max_value=200, value=30, step=5)

    st.header("Modo de execução")
    use_saved_artifacts = st.checkbox("Usar modelo/scaler salvos (.pkl)", value=True)
    retrain_inside_app = st.checkbox("Re-treinar dentro do app (mais lento)", value=False)

    st.header("Logs")
    show_logs = st.checkbox("Mostrar log", value=True)


if uploaded is None:
    st.info("Faça upload do CSV para começar.")
    st.stop()

# Salva temporariamente pra poder passar o caminho na função (mudança mínima)
tmp_path = os.path.join("tmp_upload.csv")
with open(tmp_path, "wb") as f:
    f.write(uploaded.getbuffer())

dados_formatados = carregar_dados(tmp_path)
variaveis_explicativas = dados_formatados.attrs.get("features_sugeridas")

# X e y exatamente como seu notebook
X_raw = dados_formatados[variaveis_explicativas].values
y_raw = dados_formatados["Alvo"].values

# Split temporal: últimos N para teste
indice_divisao = len(X_raw) - int(test_size_last_n)
if indice_divisao <= 0:
    st.error("Seu dataset ficou pequeno demais para esse tamanho de teste.")
    st.stop()

X_treino_raw, X_teste_raw = X_raw[:indice_divisao], X_raw[indice_divisao:]
y_treino, y_teste = y_raw[:indice_divisao], y_raw[indice_divisao:]

# Scaling SEM vazamento (igual notebook)
if use_saved_artifacts and os.path.exists(SCALER_PATH) and not retrain_inside_app:
    scaler = joblib.load(SCALER_PATH)
else:
    scaler = MinMaxScaler().fit(X_treino_raw)

X_treino = scaler.transform(X_treino_raw)
X_teste  = scaler.transform(X_teste_raw)

# Modelo
if use_saved_artifacts and os.path.exists(MODEL_PATH) and not retrain_inside_app:
    model = joblib.load(MODEL_PATH)
else:
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=10,
        grow_policy='Lossguide',
        border_count=64,
        eval_metric='F1',
        early_stopping_rounds=50,
        random_state=42,
        verbose=0
    )
    model.fit(X_treino, y_treino)

# Predições
y_pred = model.predict(X_teste)

acc = accuracy_score(y_teste, y_pred)
f1 = f1_score(y_teste, y_pred)

# Previsão do "último dia" (última linha do teste)
pred_last = int(y_pred[-1])

# Log
append_log(uploaded.name, dados_formatados, pred_last, acc, f1)

# =========================
# Dashboard
# =========================
c1, c2, c3 = st.columns([1.6, 1.0, 1.0])

with c1:
    st.subheader("Série temporal — Último")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados_formatados["Data"], y=dados_formatados["Último"], mode="lines", name="Último"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("📌 Métricas (últimos N)")
    st.metric("Acurácia (teste)", f"{acc:.2%}")
    st.metric("F1 (teste)", f"{f1:.3f}")

    if acc >= 0.75:
        st.success("✅ Acurácia ≥ 75%")
    else:
        st.warning("⚠️ Acurácia < 75% (veja dicas abaixo)")

with c3:
    st.subheader("📍 Previsão do último registro")
    if pred_last == 1:
        st.success("Tendência prevista: **ALTA (1)**")
    else:
        st.warning("Tendência prevista: **BAIXA (0)**")

st.divider()

st.subheader("Tabela: últimos N (Real vs Previsão)")
tabela = pd.DataFrame({
    "Data": dados_formatados["Data"].iloc[-int(test_size_last_n):].values,
    "Valor Real": y_teste,
    "Previsão": y_pred
})
tabela["Resultado"] = np.where(tabela["Valor Real"] == tabela["Previsão"], "✔️", "❌")
st.dataframe(tabela, use_container_width=True)

st.subheader("Matriz de confusão / Relatório")
cm = confusion_matrix(y_teste, y_pred)
st.write(cm)
st.text(classification_report(y_teste, y_pred))

st.divider()

with st.expander("Salvar artefatos (.pkl) a partir do app (opcional)"):
    if st.button("Salvar modelo_catboost.pkl e scaler_minmax.pkl"):
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        st.success("Arquivos salvos na pasta do app (útil em execução local).")

with st.expander("Dicas rápidas se a acurácia cair no app"):
    st.write(
        "- Garanta que você está usando **os mesmos .pkl** gerados no notebook.\n"
        "- Evite re-treinar no app (marque **Usar modelo/scaler salvos**).\n"
        "- Se o dataset do upload for diferente do do treino, a acurácia pode mudar.\n"
        "- O split de **últimos 30** é muito instável: tente 60/90 pra ter medida mais estável."
    )

if show_logs:
    st.subheader("🧾 Log de uso")
    ensure_log()
    log_df = pd.read_csv(LOG_PATH)
    st.dataframe(log_df.tail(50), use_container_width=True)

