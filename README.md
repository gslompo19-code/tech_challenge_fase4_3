**IBOV TrendLab — Previsão de Movimento do Ibovespa (CatBoost)**

Aplicação web em Streamlit que estima a probabilidade do IBOV subir no próximo dia (P(ALTA)) usando um modelo CatBoost previamente treinado no Tech Challenge (FIAP) — Fase 2. Nesta entrega (Fase 4), o app faz somente inferência (sem re-treino) e oferece uma experiência interativa para simulação e testes com dados próprios.
App publicado (Streamlit)

O app está online e acessível publicamente:
Link: https://techchallengefase43-fmkdk4wv8f4tdjwoyt9kay.streamlit.app/

O que o app entrega
•	P(ALTA): probabilidade estimada de o IBOV fechar mais alto no próximo dia.
•	Sinal: decisão binária baseada em um corte (Threshold).
•	P(ALTA) ≥ Threshold → ALTA
•	P(ALTA) < Threshold → BAIXA

Você pode ajustar o Threshold na barra lateral para deixar o sinal mais conservador (threshold maior) ou mais sensível (threshold menor).

**Abas do aplicativo**

1) Sandbox de Simulação (Simulação futura)
Como não existe “preço real do futuro” no dataset, esta aba permite:
•	Escolher uma data futura e um cenário.
•	Configurar retorno médio (mu), volatilidade (sigma) e seed.
•	Gerar uma trajetória simulada de preços até a data alvo.
•	Calcular P(ALTA) e Sinal ao longo do período simulado.
Importante: não é previsão de preço real futuro — é uma simulação para testar comportamento do modelo e cenários.

2) Testar com Meus Dados (Upload de CSV ou Entrada Manual)
Permite testar o modelo com dados próprios de duas formas:

2.1) Upload de CSV (histórico)
O usuário envia um CSV com histórico e o app padroniza os dados, calcula indicadores técnicos (features), permite escolher uma data e prever o dia seguinte, e plota gráfico com preço e probabilidade.
O CSV precisa conter as colunas:
•	Data (data do pregão)
•	Último (fechamento)
•	Abertura (abertura)
•	Máxima (máxima do dia)
•	Mínima (mínima do dia)
•	Vol. (volume numérico ou texto tipo 10.2M, 350K, 1.2B)

2.2) Entrada manual (um dia OHLCV)
O usuário preenche um único dia com Data, Abertura, Máxima, Mínima, Último e Volume. O app anexa esse registro ao histórico padrão para calcular as features e prever a tendência do dia seguinte ao dia inserido.

3) Histórico (dados reais do dataset)
Trabalha com dados reais do CSV padrão do projeto. Você seleciona uma data do histórico e o app calcula P(ALTA) e Sinal para o dia seguinte. As previsões usam as mesmas features utilizadas no modelo CatBoost treinado na Fase 2 do Tech Challenge (FIAP).

4) Diagnóstico (métricas do treino)
Mostra métricas fixas do treinamento (Colab / Fase 2), como acurácia (treino e teste), overfitting, F1 (CV), matriz de confusão e classification report, além de um resumo do período do dataset carregado.

Features (indicadores técnicos) usadas no modelo

O app calcula indicadores/variáveis que alimentam o modelo, incluindo:
•	Retornos: ret_1d, ret_5d, log_ret
•	Volatilidade: rv_20
•	Indicadores: rsi, macd, sinal_macd, hist_macd
•	Bollinger: bb_largura
•	ATR: atr_pct
•	Volume/OBV: vol_log, vol_ret, obv_diff
•	Z-scores: z_close_20, z_rsi_20, z_macd_20
•	Calendário: dia (dia da semana)

Há proteções contra NaN/Inf e uma correção de escala por vizinhança para lidar com valores fora de escala no histórico.
Estrutura do repositório (recomendada)
.
├── app.py
├── Dados Ibovespa (2).csv
├── modelo_catboost.pkl
├── scaler_minmax.pkl
├── requirements.txt
└── logs/
    ├── usage_log.csv
    └── usage_log.jsonl

Como rodar localmente

1) Clonar o repositório
git clone <URL_DO_SEU_REPO>
cd <PASTA_DO_REPO>

2) Criar ambiente virtual (opcional, recomendado)
Windows (PowerShell):
python -m venv .venv
.venv\Scripts\activate
Mac/Linux:
python -m venv .venv
source .venv/bin/activate

3) Instalar dependências
pip install -r requirements.txt

4) Executar o app
streamlit run app.py
requirements.txt (exemplo)
Se precisar de um modelo base:
streamlit
pandas
numpy
plotly
joblib
scikit-learn
catboost

Observação: dependendo de como o .pkl foi salvo, catboost pode ser necessário no ambiente para carregar o modelo.

Logs de uso

O app registra interações do usuário em:
•	logs/usage_log.csv
•	logs/usage_log.jsonl

Exemplos de eventos registrados: simulação futura (data alvo, cenário, mu/sigma, seed, P(ALTA), sinal), upload de CSV (nome do arquivo, linhas válidas), entrada manual OHLCV (valores e resultado), previsões no histórico e abertura do diagnóstico.

Avisos importantes
•	Projeto acadêmico/didático — não é recomendação de investimento.
•	A simulação futura usa dados simulados para criar cenário; não representa preço real do futuro.
•	O resultado do modelo depende da qualidade do histórico e das features calculadas.

Projeto

Desenvolvido para o Tech Challenge (FIAP):
•	Fase 2: treinamento do modelo (CatBoost) e validação.
•	Fase 4: aplicação em Streamlit para uso interativo (inferência, simulação e testes).
