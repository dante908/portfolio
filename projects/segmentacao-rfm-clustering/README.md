# segmentacao-rfm-clustering

Projeto de portfolio para segmentacao de clientes via RFM e clustering k-means em numpy.

## O que o projeto faz
- Gera historico de transacoes por cliente.
- Calcula RFM (`recency`, `frequency`, `monetary`).
- Aplica K-Means em NumPy para segmentar clientes.
- Nomeia personas de negocio por cluster.
- Salva base clusterizada, resumo de clusters e metadados do modelo.

## Estrutura de saida
- `data/transactions_synthetic.csv`
- `data/rfm_clusters.csv`
- `models/cluster_summary.csv`
- `models/model_info.json`
- `models/metrics.json`
- `notebooks/analysis_notes.md`

## Resultados atuais
- Numero de clusters: **5**
- Clientes segmentados: **1592**
- Maior segmento: **Potential (664 clientes)**
- Segmento Champions: **124 clientes**

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
