# recomendacao-ecommerce

Projeto de portfolio para recomendacao de produtos com feedback implicito e similaridade item-item.

## Entregas geradas na execucao
- `data/interactions_synthetic.csv`
- `data/recommendations_top10.csv`
- `models/model_info.json`
- `models/metrics.json`
- `notebooks/analysis_notes.md`

## Instalacao
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execucao
```bash
python3 src/main.py
```

## Execucao em lote (raiz do repositorio)
```bash
make run-all
```
