# 📈 IBOVESPA  — Sistema Preditivo (CatBoost + Streamlit)

Aplicação **Streamlit** para inferência (sem re-treino) de um modelo preditivo de **tendência do IBOVESPA** para o **dia seguinte** (*ALTA* / *BAIXA*), treinado previamente no Colab e empacotado como artefatos:
- `modelo_catboost.pkl` (modelo)
- `scaler_minmax.pkl` (normalizador MinMax)

O app:
1) Carrega um CSV histórico do IBOV (`Dados Ibovespa (2).csv`)  
2) Aplica um **patch de correção de escala** no preço (*Último*) para evitar o “gráfico pente”  
3) Recalcula features técnicas (RSI, MACD, Bollinger, ATR, etc.)  
4) Normaliza as features com o mesmo scaler do treinamento  
5) Gera **P(ALTA)** e sinal final com **threshold ajustável**  
6) Exibe gráficos e tabelas (histórico e simulação de cenário futuro)

---

## 🎯 Objetivo do Projeto

- **Predizer a tendência do dia seguinte** do IBOVESPA:
  - `ALTA` se `P(ALTA) >= threshold`
  - `BAIXA` caso contrário

- Entregar uma interface (Streamlit) com:
  - Consulta por data no histórico
  - Gráfico com preço + marcações dos sinais
  - Visualização de probabilidade P(ALTA)
  - Simulação futura de 30 dias (cenários)

---

## 🧠 Funcionamento do Modelo

### Alvo
O alvo é definido como:

- `1 (ALTA)` se `Último(t+1) > Último(t)`
- `0 (BAIXA)` caso contrário

A última linha do dataset é descartada por não possuir `t+1`.

### Features
O modelo utiliza exclusivamente as features abaixo:

- Retornos e volatilidade: `ret_1d`, `log_ret`, `ret_5d`, `rv_20`
- Risco e bandas: `atr_pct`, `bb_largura`, `desvio_mm3_pct`
- Volume e OBV: `vol_log`, `vol_ret`, `obv_diff`
- Indicadores técnicos: `rsi`, `macd`, `sinal_macd`, `hist_macd`
- Calendário: `dia`
- Normalização estatística: `z_close_20`, `z_rsi_20`, `z_macd_20`

As linhas com valores ausentes nessas features são removidas antes da inferência.

### Normalização
Todas as features passam pelo mesmo **MinMaxScaler** usado no treinamento:

```python
Xs = scaler.transform(X)
```

### Decisão
O modelo retorna `P(ALTA)` e o sinal final depende do **threshold** configurável:

```python
pred = (P(ALTA) >= threshold)
```

---

## 🧩 Correção de Escala do Preço

Alguns CSVs apresentam erros de escala (10x, 100x, 1000x menores).  
Para evitar o gráfico “pente”, o app aplica uma correção por vizinhança:

- Compara o preço atual com o anterior
- Ajusta multiplicando por 10, 100 ou 1000 quando necessário
- Aceita o valor quando fica próximo ao preço anterior

Esse patch roda:
- No carregamento do histórico
- Na simulação futura

---

## 🖥️ Estrutura do Streamlit

### 🧠 Aba Produto
- Seleção de data histórica
- Exibição do sinal previsto para o dia seguinte
- Gráfico interativo com:
  - Preço corrigido
  - Marcadores de ALTA/BAIXA
  - Subgráfico de P(ALTA)

### 🔮 Aba Simulação Futura
- Simulação de 30 dias com cenários:
  - Retorno constante
  - Retorno + ruído
  - Aleatório (volatilidade)
- Recalcula todas as features
- Classifica ALTA/BAIXA para cada dia simulado

⚠️ Não é previsão real, apenas **cenário hipotético**.

### 🔎 Aba Diagnóstico
- Estatísticas do dataset
- Datas inicial/final
- Verificação visual do preço corrigido

---

## 📁 Estrutura do Repositório

```
.
├── app.py
├── Dados Ibovespa (2).csv
├── modelo_catboost.pkl
├── scaler_minmax.pkl
├── requirements.txt
└── README.md
```

---

## ▶️ Execução Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🚀 Deploy no Streamlit Cloud

1. Suba o repositório no GitHub
2. Crie um novo app no Streamlit Community Cloud
3. Defina `app.py` como arquivo principal
4. Aguarde o build

---

## 🧯 Problemas Comuns

- Arquivo não encontrado: confirme nomes e caminhos
- Erro de features futuras: histórico insuficiente para janelas móveis
- Gráfico “pente”: verifique se o patch de correção está ativo

---

## ⚠️ Aviso

Projeto educacional.  
Não constitui recomendação de investimento.

---

## 📌 Próximos Passos
- Backtest com métricas financeiras
- Explainability (SHAP)
- Upload dinâmico de CSV
- Persistência de parâmetros

