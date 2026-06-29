# DocIntel

> Intelligent document processing with OCR quality assessment and validity prediction.

Trains four classifiers on synthetic document features (OCR confidence, sharpness, edge density, page count, compression ratio) to predict document validity. Dashboard provides document type composition, extraction quality metrics, and OCR confidence breakdowns.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Random Forest) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.712 |
| Gini | 0.423 |
| KS Statistic | 0.444 |
| F1 Score | 0.414 |
| Accuracy | 0.728 |

5-fold CV AUC: 0.639 ± 0.055. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Document records overview, validity distribution, feature descriptions |
| **Model Lab** | Multi-model comparison, ROC curves, calibration plots, CV results |
| **Document Analytics** | Document type composition pie chart, OCR confidence by type, page count by type |
| **Extraction Quality** | Sharpness and edge density distributions by validity, OCR confidence bin analysis |

## Repo Structure

```
DocIntel/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic document processing dataset: document type, OCR confidence, sharpness, edge density, page count, compression ratio, and validity label.

## License

MIT
