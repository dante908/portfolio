# Fraude Pagamentos RTR - Analysis Notes

Neste projeto, eu desenvolvi um pipeline para detectar transacoes com maior probabilidade de fraude em pagamentos RTR.

O que eu fiz:
- Preparei variaveis de risco de transacao e comportamento.
- Comparei opcoes e o modelo selecionado foi `rule_based`.
- Ajustei threshold para melhorar equilibrio entre detectar fraude e evitar bloqueio indevido.

Resultados principais:
- ROC-AUC teste: **0.7054**
- Recall teste: **0.3869**
- Threshold selecionado: **0.27**
- Relatorios: `score_distribution.png`, `confusion_matrix.png`, `feature_importance.png`

Como eu interpretaria isso no dia a dia:
- Transacoes com score alto vao para bloqueio/revisao prioritaria.
- O threshold deve ser revisado com base no custo de fraude e no volume operacional.
