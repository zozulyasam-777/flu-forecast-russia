# Influenza Rate Forecasting in Russia

📊 Analysis and forecasting of influenza cases per 10,000 people in the Russian Federation.

![Trend](results/flu_trend.png)

## Data
- **Source**: https://www.influenza.spb.ru/surveillance/flu-bulletin/
- **Period**: 2023–2025
- **Frequency**: Weekly
- **Geography**: Russia (by regions)

## Methods
- **ETS** (Exponential Smoothing)
- **Prophet** (by Meta)
- **SARIMA**

## Results
3-month forecast with account for **yearly seasonality**.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
