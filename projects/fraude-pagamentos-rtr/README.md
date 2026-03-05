# fraude-pagamentos-rtr

Projeto de portfolio para deteccao de fraude em pagamentos RTR com score de risco e threshold tuning.

## Entregas geradas na execucao
- `data/transactions_synthetic.csv`
- `data/transactions_test_scored.csv`
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
