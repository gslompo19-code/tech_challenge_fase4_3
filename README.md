📈 IBOVESPA — Sistema Preditivo (CatBoost + Streamlit)

Aplicação Streamlit para inferência (sem re-treino) de um modelo preditivo de tendência do IBOVESPA para o dia seguinte (ALTA / BAIXA), treinado previamente no Google Colab e empacotado como artefatos:

•	modelo_catboost.pkl — modelo CatBoostClassifier
•	scaler_minmax.pkl — normalizador MinMaxScaler

O app foi desenvolvido como entrega do Tech Challenge – Fase 4, com foco em deploy, monitoramento e visualização do modelo.

🎯 Objetivo do Projeto

Predizer a tendência do IBOVESPA no dia seguinte, a partir de dados históricos:

•	ALTA se P(ALTA) ≥ threshold
•	BAIXA caso contrário

Disponibilizar uma interface interativa em Streamlit que permita:

•	Consultar previsões por data histórica
•	Visualizar sinais do modelo no tempo
•	Ajustar o threshold de decisão
•	Avaliar métricas fixas do modelo
•	Executar backtest completo no histórico
•	Registrar logs de uso (simulação de produção)

🧠 Funcionamento do Modelo

Alvo
O alvo é definido como:
•	1 (ALTA) se Último(t+1) > Último(t)
•	0 (BAIXA) caso contrário

A última linha do dataset é descartada por não possuir o valor de t+1.

Features Utilizadas

Retorno e volatilidade:
•	ret_1d, log_ret, ret_5d, rv_20

Risco e bandas:
•	atr_pct, bb_largura, desvio_mm3_pct

Volume e fluxo:
•	vol_log, vol_ret, obv_diff

Indicadores técnicos:
•	rsi, macd, sinal_macd, hist_macd

Calendário:
•	dia

Normalização estatística:
•	z_close_20, z_rsi_20, z_macd_20

Linhas com valores ausentes nessas features são removidas antes da inferência.

Normalização

As features são normalizadas com o mesmo MinMaxScaler usado no treinamento:
Xs = scaler.transform(X)

Decisão

O modelo retorna P(ALTA) e o sinal final depende de um threshold ajustável:

pred = (P(ALTA) >= threshold)

🧩 Correção de Escala do Preço (Patch Anti “Gráfico Pente”)

Alguns CSVs históricos apresentam erros de escala no preço (Último), com valores 10x, 100x ou 1000x menores. Para evitar distorções visuais, o app aplica uma correção automática por vizinhança:

•	Compara o preço atual com o dia anterior
•	Testa fatores de correção (10, 100, 1000)
•	Ajusta quando o valor corrigido fica próximo ao preço anterior

Esse patch é aplicado:

•	No carregamento do histórico
•	Antes da geração de gráficos e sinais

🖥️ Estrutura do Streamlit

🧠 Aba Produto
•	Seleção de data histórica
•	Previsão da tendência do dia seguinte
•	Exibição de P(ALTA), P(BAIXA) e sinal final
•	Gráfico interativo com preço corrigido, sinais e probabilidade
•	Registro automático ou manual de logs de uso

📉 Aba Backtest Completo (Opcional)
•	Predição em todo o histórico disponível
•	Comparação entre previsão e alvo real
•	Acurácia no histórico completo
•	Gráfico observado vs previsto
•	Download do backtest em CSV
O modelo não é re-treinado, tratando-se apenas de inferência retrospectiva.

🔎 Aba Diagnóstico
•	Acurácia de treino e teste
•	F1-score médio (cross-validation)
•	Overfitting
•	Matriz de confusão
•	Classification report
•	Auditoria do dataset
•	Visualização e download do log de uso

⚠️ Aviso Legal
Projeto estritamente educacional. Não constitui recomendação de investimento ou aconselhamento financeiro.

📌 Próximos Passos (Evolução)
•	Métricas financeiras (retorno acumulado, drawdown)
•	Explainability (SHAP)
•	Upload dinâmico de CSV pelo usuário
•	Monitoramento contínuo de drift
•	Persistência de parâmetros do usuário
