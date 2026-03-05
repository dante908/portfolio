# segmentacao-rfm-clustering

Projeto de portfolio para segmentacao de clientes via RFM e clustering k-means em numpy.

## Entregas geradas na execucao
- `data/transactions_synthetic.csv`
- `data/rfm_clusters.csv`
- `models/cluster_summary.csv`
- `models/model_info.json`
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
