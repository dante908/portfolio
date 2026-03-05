# Recomendacao Ecommerce - Analysis Notes

Neste projeto, eu desenvolvi um sistema de recomendacao para sugerir produtos com maior chance de interesse do usuario.

O que eu fiz:
- Modelei interacoes implicitas e gerei ranking top-10 por usuario.
- Avaliei desempenho com HitRate@10 e MRR@10.
- Comparei baseline com opcao avancada.

Resultados principais:
- Modelo selecionado: **item_item_cosine**
- HitRate@10: **0.1333**
- MRR@10: **0.0531**
- Baseline HitRate@10: **0.1333**
- Modelo avancado HitRate@10: **None**
- Usuarios avaliados: **450**
- Relatorios: `event_distribution.png`, `rank_coverage.png`, `top_recommended_items.png`

Como eu interpretaria isso no dia a dia:
- O baseline atual ja gera recomendacoes funcionais para comeco.
- O proximo passo e testar em experimento para medir impacto real em conversao.
