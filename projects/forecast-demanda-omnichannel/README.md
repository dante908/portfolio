# forecast-demanda-omnichannel

Projeto de portfolio para previsao de demanda por canal e SKU em contexto omnichannel.

## Entregas geradas na execucao
- `data/demand_history_synthetic.csv`
- `data/forecast_backtest.csv`
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
