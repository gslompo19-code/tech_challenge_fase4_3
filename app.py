# app.py — IBOV Signal (versão final)
# ✅ log (bônus) + leitura segura (sem ParserError)
# ✅ backtest completo opcional (10 anos) sem Colab
# ✅ melhorias de UX na aba Produto
# Obs: este app NÃO retreina o modelo. Ele apenas carrega model + scaler treinados no Colab.

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

DEFAULT_CSV = "Dados Ibovespa (2).csv"   # coloque este arquivo no repositório com este nome (ou ajuste o path)
MODEL_PATH = "modelo_catboost.pkl"
SCALER_PATH = "scaler_minmax.pkl"

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "uso_app.csv")  # salvamos com sep=";" (mais robusto)


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
    "cm": [[13, 3],
           [3, 11]],
    "report": """precision    recall  f1-score   support

0       0.81      0.81      0.81        16
1       0.79      0.79      0.79        14

accuracy                           0.80        30
macro avg       0.80      0.80      0.80        30
weighted avg    0.80      0.80      0.80        30"""
}


# =========================
# LOG (bônus): escrita + leitura segura
# =========================
def append_log_csv(path: str, row_df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    row_df.to_csv(
        path,
        mode="a",
        header=header,
        index=False,
        sep=";",
        encoding="utf-8"
    )


def read_log_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()

    # 1) tenta o formato “oficial” do app (sep=";")
    try:
        return pd.read_csv(path, sep=";", engine="python", on_bad_lines="skip")
    except Exception:
        pass

    # 2) fallback: se o arquivo estiver com vírgula por alguma versão antiga
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


# =========================
# Funções do Colab (feature engineering)
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
    Corrige 'Último' quando alguns pontos vêm 10x/100x/1000x menores (CSV mal formatado).
    """
    df = df.copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df["Último"] = pd.to_numeric(df["Último"], errors="coerce")
    df = df.dropna(subset=["Data", "Último"]).sort_values("Data").reset_index(drop=True)

    for i in range(1, len(df)):
        prev = df.loc[i - 1, "Último"]
        curr = df.loc[i, "Último"]

        # se caiu mais de 80% de um dia pro outro, provavelmente escala errada
        if curr < prev * 0.2:
            for fator in [10, 100, 1000]:
                if prev * 0.7 < curr * fator < prev * 1.3:
                    df.loc[i, "Último"] = curr * fator
                    break

    return df


def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    """
    Pipeline do Colab + patch de correção de escala
    """
    df = pd.read_csv(caminho_csv)
    df.columns = df.columns.str.strip()

    # Data
    df["Data"] = pd.to_datetime(df["Data"], format="%d.%m.%Y", errors="coerce")
    df = df.sort_values("Data").dropna(subset=["Data"])

    # Volume
    df["Vol."] = df["Vol."].apply(volume_to_float)

    # Preços BR -> float
    for coluna in ["Último", "Abertura", "Máxima", "Mínima"]:
        df[coluna] = (
            df[coluna].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    # Patch de escala (corrigir “pente”)
    df = correcao_escala_por_vizinhanca(df)
    df = df.sort_values("Data").reset_index(drop=True)

    # Features
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

    # Bollinger (20)
    bb_media = df["Último"].rolling(20, min_periods=20).mean()
    bb_std = df["Último"].rolling(20, min_periods=20).std()
    df["bb_media"] = bb_media
    df["bb_std"] = bb_std
    df["bb_sup"] = bb_media + 2 * bb_std
    df["bb_inf"] = bb_media - 2 * bb_std
    df["bb_largura"] = (df["bb_sup"] - df["bb_inf"]) / bb_media

    # ATR (14)
    tr1 = df["Máxima"] - df["Mínima"]
    tr2 = (df["Máxima"] - df["Último"].shift(1)).abs()
    tr3 = (df["Mínima"] - df["Último"].shift(1)).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(14, min_periods=14).mean()

    # OBV + Alvo
    df["obv"] = obv_series(df)
    df["Alvo"] = (df["Último"].shift(-1) > df["Último"]).astype("int8")
    df = df.iloc[:-1].copy()  # remove último sem alvo

    # Transformações
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
# Predição
# =========================
def predict_proba_batch(model, scaler, X, threshold):
    Xs = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)[:, 1]
    else:
        # fallback
        proba = model.predict(Xs).astype(float)
        proba = np.clip(proba, 0.0, 1.0)
    pred = (proba >= threshold).astype(int)
    return pred, proba


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
            mode="markers", name="ALTA",
            marker=dict(size=9, symbol="triangle-up")
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=np.where(pred == 0, price_vals, np.nan),
            mode="markers", name="BAIXA",
            marker=dict(size=8, symbol="triangle-down")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=dates, y=proba, mode="lines", fill="tozeroy", name="P(ALTA)"),
        row=2, col=1
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_width=2,
        annotation_text=f"threshold={threshold:.2f}",
        row=2, col=1
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
# Backtest completo (opcional)
# =========================
# ✅ CORREÇÃO: não passamos model/scaler em função cacheada (dá UnhashableParamError).
# Pegamos model+scaler de dentro, via load_model_and_scaler() que está em cache_resource.
@st.cache_data(show_spinner="Gerando backtest completo...")
def compute_backtest_full(df_feat: pd.DataFrame, features: list, threshold: float) -> pd.DataFrame:
    model, scaler = load_model_and_scaler()

    X_all = df_feat[features].values
    y_all = df_feat["Alvo"].values
    pred_all, proba_all = predict_proba_batch(model, scaler, X_all, threshold)

    out = pd.DataFrame({
        "Data": df_feat["Data"].values,
        "Preço": df_feat["Último"].values,
        "Alvo_real": y_all,
        "Pred": pred_all,
        "P_ALTA": proba_all
    })
    out["Acerto"] = np.where(out["Alvo_real"] == out["Pred"], "✔️", "❌")
    out["Sinal"] = np.where(out["Pred"] == 1, "ALTA", "BAIXA")
    return out


def make_backtest_scatter(bt: pd.DataFrame) -> go.Figure:
    # Pontinhos com duas cores: observado vs previsto (classes)
    # Para “observado” e “previsto”, usamos y=0/1 (queda/alta), com jitter leve só para visualizar
    bt = bt.copy()
    bt["y_obs"] = bt["Alvo_real"] + np.random.normal(0, 0.02, size=len(bt))
    bt["y_pred"] = bt["Pred"] + np.random.normal(0, 0.02, size=len(bt))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bt["Data"], y=bt["y_obs"],
        mode="markers",
        name="Observado (real)",
        marker=dict(size=6, symbol="circle"),
        customdata=np.stack([bt["Preço"], bt["Alvo_real"], bt["Pred"], bt["P_ALTA"], bt["Acerto"]], axis=1),
        hovertemplate=(
            "Data=%{x|%Y-%m-%d}<br>"
            "Preço=%{customdata[0]:.2f}<br>"
            "Real=%{customdata[1]:.0f} / Pred=%{customdata[2]:.0f}<br>"
            "P(ALTA)=%{customdata[3]:.2f}<br>"
            "Acerto=%{customdata[4]}<extra></extra>"
        )
    ))

    fig.add_trace(go.Scatter(
        x=bt["Data"], y=bt["y_pred"],
        mode="markers",
        name="Previsto (modelo)",
        marker=dict(size=6, symbol="diamond"),
        customdata=np.stack([bt["Preço"], bt["Alvo_real"], bt["Pred"], bt["P_ALTA"], bt["Acerto"]], axis=1),
        hovertemplate=(
            "Data=%{x|%Y-%m-%d}<br>"
            "Preço=%{customdata[0]:.2f}<br>"
            "Real=%{customdata[1]:.0f} / Pred=%{customdata[2]:.0f}<br>"
            "P(ALTA)=%{customdata[3]:.2f}<br>"
            "Acerto=%{customdata[4]}<extra></extra>"
        )
    ))

    fig.update_yaxes(
        title="Classe (0=Queda, 1=Alta)",
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["Queda (0)", "Alta (1)"],
        range=[-0.3, 1.3]
    )
    fig.update_layout(
        title="Backtest (período completo) — Observado vs Previsto (pontinhos)",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h")
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


# =========================
# App
# =========================
st.title("📈 IBOV Signal — Sistema Preditivo (modelo do Colab, sem re-treino)")

with st.sidebar:
    st.header("Config")
    threshold = st.slider("Threshold para ALTA", 0.30, 0.70, 0.50, 0.01, key="sb_threshold")
    view_n = st.slider("Janela do gráfico (últimos N)", 60, 1500, 400, 20, key="sb_view_n")
    st.caption("Patch de escala do `Último` aplicado (corrige gráfico 'pente').")

    st.divider()
    st.subheader("Log (bônus)")
    st.caption("O app salva as previsões registradas em logs/uso_app.csv")

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

tab_produto, tab_backtest, tab_diag = st.tabs([
    "🧠 Produto",
    "📉 Backtest completo (opcional)",
    "🔎 Diagnóstico (métricas + log)"
])

# =========================
# ABA 1 — Produto (UX melhorada)
# =========================
with tab_produto:
    st.subheader("Produto: selecione uma data e obtenha a tendência do dia seguinte")

    # UX: filtros de data + busca rápida
    min_date = df["Data"].min().date()
    max_date = df["Data"].max().date()

    colL, colM, colR = st.columns([1.2, 1.2, 1.6])
    with colL:
        selected_date = st.date_input(
            "Data (histórico)",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="prod_date"
        )
    with colM:
        st.caption("Dica: a previsão refere-se ao **dia seguinte** ao selecionado.")
        show_feature_box = st.checkbox("Mostrar detalhes das features do dia", value=True, key="prod_show_features")
    with colR:
        st.markdown("**Ações rápidas**")
        register_log_auto = st.checkbox("Registrar no log automaticamente ao prever", value=True, key="prod_auto_log")
        st.caption("Se desativar, use o botão “Registrar no log” após prever.")

    matches = df.index[df["Data"].dt.date == selected_date].tolist()
    if not matches:
        st.warning("Data não encontrada no histórico processado (pode ter sido filtrada por NaN nas features).")
        st.stop()

    idx = matches[0]

    X_sel = df.loc[[idx], features].values
    pred_sel, proba_sel = predict_proba_batch(model, scaler, X_sel, threshold)
    y = int(pred_sel[0])
    p_alta = float(proba_sel[0])
    p_baixa = 1.0 - p_alta

    # UX: cards de probabilidade + decisão
    c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
    with c1:
        st.metric("📈 P(ALTA)", f"{p_alta:.2%}")
    with c2:
        st.metric("📉 P(BAIXA)", f"{p_baixa:.2%}")
    with c3:
        if y == 1:
            st.success(f"📈 Tendência prevista (dia seguinte): **ALTA** — confiança {p_alta:.2%}")
        else:
            st.warning(f"📉 Tendência prevista (dia seguinte): **BAIXA** — confiança {p_baixa:.2%}")

        st.progress(int(p_alta * 100))
        st.caption("A barra representa a probabilidade estimada de **ALTA**.")

    # Detalhe do dia
    if show_feature_box:
        st.write("Registro do dia selecionado (amostra de indicadores):")
        cols_show = ["Data", "Último", "Vol.", "rsi", "macd", "bb_largura", "atr_pct", "Alvo"]
        cols_show = [c for c in cols_show if c in df.columns]
        st.dataframe(df.loc[[idx], cols_show], use_container_width=True)

    # LOG: registrar previsão
    log_row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now().isoformat(),
        "selected_date": str(selected_date),
        "threshold": float(threshold),
        "p_alta": float(p_alta),
        "p_baixa": float(p_baixa),
        "pred": int(y),
        "preco_ultimo": float(df.loc[idx, "Último"]),
        "rsi": float(df.loc[idx, "rsi"]) if pd.notna(df.loc[idx, "rsi"]) else np.nan,
        "macd": float(df.loc[idx, "macd"]) if pd.notna(df.loc[idx, "macd"]) else np.nan,
        "bb_largura": float(df.loc[idx, "bb_largura"]) if pd.notna(df.loc[idx, "bb_largura"]) else np.nan,
        "atr_pct": float(df.loc[idx, "atr_pct"]) if pd.notna(df.loc[idx, "atr_pct"]) else np.nan,
    }])

    btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1.2, 2.6])
    with btn_col1:
        if st.button("📝 Registrar no log", key="prod_btn_log"):
            append_log_csv(LOG_PATH, log_row)
            st.success("Previsão registrada em logs/uso_app.csv")
    with btn_col2:
        if st.button("🔄 Atualizar gráfico", key="prod_btn_refresh"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    with btn_col3:
        st.caption("O gráfico abaixo mostra o histórico + sinais do modelo (últimos N conforme a barra lateral).")

    # Auto-log
    if register_log_auto:
        if "last_logged_key" not in st.session_state:
            st.session_state["last_logged_key"] = None

        current_key = (str(selected_date), float(threshold), int(y), round(p_alta, 6))
        if st.session_state["last_logged_key"] != current_key:
            append_log_csv(LOG_PATH, log_row)
            st.session_state["last_logged_key"] = current_key

    # Gráfico histórico + sinais
    df_plot = df.tail(int(view_n)).copy()
    X_plot = df_plot[features].values
    pred_plot, proba_plot = predict_proba_batch(model, scaler, X_plot, threshold)

    fig = make_signal_chart(
        df_plot,
        pred_plot,
        proba_plot,
        threshold,
        "Histórico + Sinais do modelo (com correção de escala)"
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================
# ABA 2 — Backtest completo opcional (10 anos)
# =========================
with tab_backtest:
    st.subheader("📉 Backtest completo (opcional) — período inteiro do CSV")

    st.write(
        "Aqui o app calcula a predição para **todo o histórico processado** e compara com o alvo real (dia seguinte). "
        "Isso elimina a necessidade de ajustar o Colab só para exibir o período inteiro."
    )

    colA, colB = st.columns([1.1, 1.9])
    with colA:
        run_bt = st.button("▶️ Gerar Backtest completo", key="bt_run")
        max_rows = st.slider("Máx. linhas na tabela", 50, 2000, 300, 50, key="bt_max_rows")
    with colB:
        st.caption("O cálculo é cacheado. Se mudar o threshold, gere novamente.")

    if run_bt:
        # ✅ CORREÇÃO: não passamos model/scaler para a função cacheada
        bt_full = compute_backtest_full(df, features, threshold)

        # Métricas rápidas (no histórico inteiro)
        acc_full = (bt_full["Alvo_real"] == bt_full["Pred"]).mean()
        st.metric("Acurácia (histórico completo)", f"{acc_full:.2%}")

        fig_scatter = make_backtest_scatter(bt_full)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.write("Amostra do backtest (últimos registros):")
        st.dataframe(bt_full.tail(int(max_rows)), use_container_width=True)

        st.download_button(
            "⬇️ Baixar backtest completo (CSV)",
            data=bt_full.to_csv(index=False, sep=";").encode("utf-8"),
            file_name="backtest_completo.csv",
            mime="text/csv",
            key="bt_download"
        )
    else:
        st.info("Clique em **Gerar Backtest completo** para processar todo o período.")


# =========================
# ABA 3 — Diagnóstico (métricas + log)
# =========================
with tab_diag:
    st.subheader("🔎 Diagnóstico (métricas fixas do Colab — sem re-treino)")

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
    st.subheader("📄 Log de uso (bônus) — leitura segura")

    log_df = read_log_csv_safe(LOG_PATH)
    if len(log_df) == 0:
        st.info("Nenhum log encontrado ainda. Gere previsões na aba Produto (o auto-log pode estar ligado).")
    else:
        st.write("Últimos 50 registros do log:")
        st.dataframe(log_df.tail(50), use_container_width=True)

        st.download_button(
            "⬇️ Baixar log (CSV)",
            data=log_df.to_csv(index=False, sep=";").encode("utf-8"),
            file_name="uso_app.csv",
            mime="text/csv",
            key="log_download"
        )

    st.divider()
    st.subheader("Auditoria do dataset carregado")

    d1, d2, d3 = st.columns(3)
    d1.metric("Linhas válidas (features)", len(df))
    d2.metric("Data inicial", str(df["Data"].iloc[0].date()))
    d3.metric("Data final", str(df["Data"].iloc[-1].date()))

    st.write("Resumo do `Último` (corrigido):")
    st.write(df["Último"].describe())

    st.write("Últimos 10 pontos (Data, Último):")
    st.dataframe(df[["Data", "Último"]].tail(10), use_container_width=True)
