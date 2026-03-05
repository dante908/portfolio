# Forecast Demanda Omnichannel - Analysis Notes

Neste projeto, eu construí um pipeline de previsao de demanda para apoiar reposicao e planejamento de estoque.

O que eu fiz:
- Estruturei a serie temporal com variaveis de calendario e contexto.
- Treinei o modelo `ridge_numpy` e comparei com baseline (`lag7`).
- Deixei o fluxo preparado para rodar de forma recorrente.

Resultados principais:
- MAPE validacao (modelo): **6.0631%**
- MAPE teste (modelo): **5.9667%**
- MAPE teste (baseline lag7): **9.0547%**
- Relatorios: `daily_actual_vs_forecast.png`, `abs_error_distribution.png`, `top_series_actual_demand.png`

Como eu interpretaria isso no dia a dia:
- O modelo superou o baseline e pode apoiar decisao de estoque com mais confianca.
- O ideal e acompanhar erro por SKU e periodo para ajustes continuos.
