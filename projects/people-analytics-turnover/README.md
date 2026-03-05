# people-analytics-turnover

Projeto de portfolio para prever turnover de colaboradores e apoiar acoes de retencao.

## O que o projeto faz
- Gera base de colaboradores com sinais de risco (engajamento, horas extras, faltas, salario e gestao).
- Treina regressao logistica em NumPy.
- Ajusta threshold por validacao.
- Avalia ROC-AUC, accuracy, precision, recall e F1.
- Salva base de teste com score de turnover e artefatos do modelo.

## Estrutura de saida
- `data/employees_synthetic.csv`
- `data/employees_test_scored.csv`
- `models/model.json`
- `models/metrics.json`
- `notebooks/analysis_notes.md`

## Resultados atuais
- ROC-AUC teste: **0.6355**
- F1 teste: **0.2365**
- Recall teste: **0.4211**
- Threshold selecionado: **0.58**

## Instalacao minima
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Como reproduzir
```bash
python3 -m pip install -r requirements.txt
python3 src/main.py
```

## Execucao em lote (raiz do repositorio)
```bash
make run-all
```
