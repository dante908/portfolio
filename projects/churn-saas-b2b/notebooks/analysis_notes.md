# Churn SaaS B2B - Analysis Notes

Neste projeto, eu montei um pipeline para estimar churn de clientes SaaS e priorizar acao do time de CS.

O que eu fiz:
- Gerei a base e preparei as variaveis mais importantes de comportamento e risco.
- Treinei e comparei abordagens, e o modelo selecionado foi `rule_based`.
- Ajustei o threshold para equilibrar acerto e acao operacional.

Resultados principais:
- ROC-AUC (teste): **0.704**
- F1-score (teste): **0.6688**
- Threshold selecionado: **0.37**
- Relatorios: `score_distribution.png`, `confusion_matrix.png`, `feature_importance.png`

Como eu interpretaria isso no dia a dia:
- Contas com score alto entram primeiro na fila de contato.
- O threshold pode ser recalibrado conforme capacidade do time e custo de retencao.
