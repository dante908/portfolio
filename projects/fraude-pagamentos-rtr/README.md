# fraude-pagamentos-rtr

Projeto de portfolio para deteccao de fraude em pagamentos RTR com score de risco e threshold tuning.

## O que o projeto faz
- Gera base de transacoes com variaveis de risco (amount, velocity, device/merchant risk e internacional).
- Cria score de risco por regras de negocio.
- Ajusta threshold por F1 na validacao.
- Avalia ROC-AUC, precision, recall, F1 e accuracy.
- Salva base de teste com score/predicao e metricas finais.

## Estrutura de saida
- `data/transactions_synthetic.csv`
- `data/transactions_test_scored.csv`
- `models/model_info.json`
- `models/metrics.json`
- `notebooks/analysis_notes.md`

## Resultados atuais
- ROC-AUC teste: **0.7083**
- F1 teste: **0.3427**
- Recall teste: **0.5455**
- Threshold selecionado: **0.25**

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
