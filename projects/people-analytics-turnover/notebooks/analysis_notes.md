# People Analytics Turnover - Analysis Notes

Neste projeto, eu criei um modelo para sinalizar risco de turnover e apoiar a priorizacao de acoes de retencao.

O que eu fiz:
- Trabalhei variaveis de contexto de colaborador, engajamento e historico.
- Treinei e validei o modelo `logistic_regression`.
- Ajustei threshold para buscar melhor F1 na validacao.

Resultados principais:
- ROC-AUC teste: **0.6355**
- F1 teste: **0.2365**
- Recall teste: **0.4211**
- Threshold selecionado: **0.58**
- ROC-AUC validacao (logreg): **0.6495**
- ROC-AUC validacao (xgboost): **None**
- Relatorios: `score_distribution.png`, `confusion_matrix.png`, `feature_importance.png`

Como eu interpretaria isso no dia a dia:
- O score ajuda a priorizar acompanhamento de grupos com maior risco.
- O resultado apoia decisao humana e deve ser usado com criterio de RH.
