# churn-saas-b2b

Projeto de portfolio para predicao de churn em SaaS B2B com pipeline completo e reproduzivel.

## O que o projeto faz
- Gera dataset estruturado de contas B2B para modelagem de churn.
- Treina baseline de classificacao (logistic regression em numpy) e compara com baseline rule-based.
- Avalia ROC-AUC, accuracy, precision, recall e F1.
- Salva base, score de teste, modelo e metricas.

## Estrutura de saida
- `data/churn_saas_synthetic.csv`
- `data/churn_test_scored.csv`
- `models/churn_model.json`
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
