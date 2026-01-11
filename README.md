
📈 IBOVESPA — Sistema Preditivo (CatBoost + Streamlit)

Aplicação Streamlit para inferência (sem re-treino) de um modelo preditivo de tendência do IBOVESPA para o dia seguinte (ALTA / BAIXA), treinado previamente no Google Colab e empacotado como artefatos:

modelo_catboost.pkl — modelo CatBoostClassifier

scaler_minmax.pkl — normalizador MinMaxScaler

O app foi desenvolvido como entrega do Tech Challenge – Fase 4, com foco em deploy, monitoramento e visualização do modelo.

🎯 Objetivo do Projeto

Predizer a tendência do IBOVESPA no dia seguinte, a partir de dados históricos:

ALTA se P(ALTA) ≥ threshold

BAIXA caso contrário

Disponibilizar uma interface interativa em Streamlit que permita:

Consultar previsões por data histórica

Visualizar sinais do modelo no tempo

Ajustar o threshold de decisão

Avaliar métricas fixas do modelo

Executar backtest completo no histórico

Registrar logs de uso (simulação de produção)

🧠 Funcionamento do Modelo
Alvo

O alvo é definido como:

1 (ALTA) se Último(t+1) > Último(t)

0 (BAIXA) caso contrário

A última linha do dataset é descartada por não possuir o valor de t+1.

Features Utilizadas

O modelo utiliza exclusivamente as seguintes features técnicas:

Retorno e volatilidade

ret_1d, log_ret, ret_5d, rv_20

Risco e bandas

atr_pct, bb_largura, desvio_mm3_pct

Volume e fluxo

vol_log, vol_ret, obv_diff

Indicadores técnicos

rsi, macd, sinal_macd, hist_macd

Calendário

dia

Normalização estatística

z_close_20, z_rsi_20, z_macd_20

Linhas com valores ausentes nessas features são removidas antes da inferência.

Normalização

As features são normalizadas com o mesmo MinMaxScaler usado no treinamento:

Xs = scaler.transform(X)

Decisão

O modelo retorna P(ALTA) e o sinal final depende de um threshold ajustável:

pred = (P(ALTA) >= threshold)

🧩 Correção de Escala do Preço (Patch Anti “Gráfico Pente”)

Alguns CSVs históricos apresentam erros de escala no preço (Último), com valores 10x, 100x ou 1000x menores.

Para evitar distorções visuais, o app aplica uma correção automática por vizinhança:

Compara o preço atual com o dia anterior

Testa fatores de correção (10, 100, 1000)

Ajusta quando o valor corrigido fica próximo ao preço anterior

Esse patch é aplicado:

No carregamento do histórico

Antes da geração de gráficos e sinais

🖥️ Estrutura do Streamlit
🧠 Aba Produto

Interface principal do sistema:

Seleção de data histórica

Previsão da tendência do dia seguinte

Exibição de:

P(ALTA)

P(BAIXA)

Sinal final (ALTA / BAIXA)

Gráfico interativo com:

Preço corrigido

Marcações de sinais do modelo

Subgráfico da probabilidade P(ALTA)

Registro automático ou manual de logs de uso

📉 Aba Backtest Completo (Opcional)

Executa a predição em todo o histórico disponível

Compara previsão vs alvo real

Exibe:

Acurácia no histórico completo

Gráfico de dispersão (observado vs previsto)

Tabela com resultados

Permite download do backtest em CSV

⚠️ O modelo não é re-treinado — trata-se apenas de inferência retrospectiva.

🔎 Aba Diagnóstico

Painel fixo de métricas do modelo (obtidas no Colab):

Acurácia de treino e teste

F1-score médio (cross-validation)

Overfitting

Matriz de confusão (tabela + gráfico)

Classification report

Auditoria do dataset carregado

Visualização e download do log de uso

📁 Estrutura do Repositório
.
├── app.py
├── Dados Ibovespa (2).csv
├── modelo_catboost.pkl
├── scaler_minmax.pkl
├── requirements.txt
└── README.md

▶️ Execução Local
pip install -r requirements.txt
streamlit run app.py

🚀 Deploy no Streamlit Cloud

Suba o repositório no GitHub

Crie um novo app no Streamlit Community Cloud

Defina app.py como arquivo principal

Aguarde o build e publique

🧯 Problemas Comuns

Arquivo não encontrado
Verifique nomes e caminhos do CSV e dos arquivos .pkl.

Dataset pequeno
Pode não haver dados suficientes para janelas móveis (RSI, Bollinger, etc.).

Gráfico distorcido
Confirme que o patch de correção de escala está ativo.

⚠️ Aviso Legal

Projeto estritamente educacional.
Não constitui recomendação de investimento ou aconselhamento financeiro.

📌 Próximos Passos (Evolução)

Métricas financeiras (retorno acumulado, drawdown)

Explainability (SHAP)

Upload dinâmico de CSV pelo usuário

Monitoramento contínuo de drift

Persistência de parâmetros do usuário
