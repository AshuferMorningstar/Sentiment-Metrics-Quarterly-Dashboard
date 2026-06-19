# Sentiment Metrics Quarterly Dashboard

This repository contains a small data-cleaning pipeline for the Sentiment140 dataset. The main script downloads the dataset with `kagglehub`, cleans and enriches the text data, and writes dashboard-ready CSV outputs.

## What It Does

- Downloads the Sentiment140 dataset from Kaggle.
- Loads a sample of 5,000 rows for a quick demo run.
- Removes duplicate rows and rows with missing text.
- Adds `sentiment_label`, `text_length`, and `word_count` features.
- Aggregates sentiment metrics for reporting.
- Exports `clean_sentiment.csv` and `metrics.csv`.

## Requirements

- Python 3.10 or newer
- `pandas`
- `kagglehub`

Install the dependencies with:

```bash
pip install pandas kagglehub
```

## Usage

Run the cleaning script from the project root:

```bash
python 1_data_cleaning.py
```

The script will download the dataset on first run, clean the data, and print a completeness percentage plus the sentiment summary table.

## Output Files

- `clean_sentiment.csv`: cleaned row-level data with the derived features.
- `metrics.csv`: grouped sentiment metrics for the dashboard.

## Notes

- The raw Sentiment140 CSV is downloaded locally during execution and should not be committed to the repository.
- `metrics.csv` is the intended tracked CSV output.
- If you want to change the sample size, update the `nrows=5000` value in `1_data_cleaning.py`.