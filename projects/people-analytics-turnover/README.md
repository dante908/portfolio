# people-analytics-turnover

Projeto de portfolio para prever turnover de colaboradores e apoiar acoes de retencao.

## Entregas geradas na execucao
- `data/employees_synthetic.csv`
- `data/employees_test_scored.csv`
- `models/model.json`
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
