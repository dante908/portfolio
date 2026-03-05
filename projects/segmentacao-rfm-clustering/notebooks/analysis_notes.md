# Segmentacao RFM - Analysis Notes

Neste projeto, eu segmentei clientes por comportamento de compra usando RFM + clustering.

O que eu fiz:
- Calculei Recency, Frequency e Monetary por cliente.
- Testei candidatos de K e selecionei o melhor por silhouette.
- Associei clusters a personas para facilitar acao comercial.

Resultados principais:
- K selecionado: **4**
- Silhouette: **0.4061**
- Clientes segmentados: **1592**
- Maior segmento: **Loyal (778 clientes)**
- Champions: **238 clientes**
- Relatorios: `cluster_sizes.png`, `frequency_monetary_scatter.png`, `k_selection_silhouette.png`

Como eu interpretaria isso no dia a dia:
- Cada persona pode receber campanha e estrategia diferente.
- Recalcular periodicamente ajuda a manter segmentacao atualizada.
