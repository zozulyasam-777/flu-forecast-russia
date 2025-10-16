# main.py
"""
Prediction of influenza incidence in Russia.
Author: Sam Zozulya
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def load_data(data_dir: str) -> pd.DataFrame:
    """
    Downloads and connects data from CSV files for 2023-2025.

    Args:
        data_dir (str): The Data Folder Path.

    Returns:
        pd.DataFrame: United DataFrame with columns 'date', 'rate', 'year'
    """
    files = {
        2023: 'stat_flu_2023.csv',
        2024: 'stat_flu_2024.csv',
        2025: 'stat_flu_2025.csv'
    }

    df_list = []

    for year, filename in files.items():
        path = os.path.join(data_dir, filename)

        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue

        # Reading: week, rate_per_10k
        df = pd.read_csv(path, header=None, names=['week', 'rate'])
        df['year'] = year

        # Type check
        df['week'] = pd.to_numeric(df['week'], errors='coerce')
        df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
        df.dropna(inplace=True)
        df = df[(df['week'] >= 1) & (df['week'] <= 53)]

        # Convert week to date
        df['date'] = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta((df['week'] - 1) * 7, unit='D')

        df_list.append(df[['date', 'rate', 'year']])

    if not df_list:
        raise FileNotFoundError("Failed to load any files. Check the folder! 'data/'")

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.sort_values('date', inplace=True)
    return combined_df


def forecast_ets(df_hist: pd.DataFrame, steps: int = 3) -> pd.DataFrame:
    """
    predicts using ETS's method.

    Args:
        df_hist (pd.DataFrame): Historical data before forecast point
        steps (int): Number of steps (weeks) for forecast

    Returns:
        pd.DataFrame: Forecast (future only)
    """
    series = df_hist.set_index('date')['rate'].sort_index()

    if len(series) < 10:
        raise ValueError("Too little data to simulate")

    MIN_SEASONAL_LENGTH = 104 # Threshold: minimum 2 full seasons (104 weeks) for seasonal ETS
    # Checking if there is enough data for the seasonal model
    if len(series) >= MIN_SEASONAL_LENGTH:
        model = ExponentialSmoothing(
            series,
            trend='add',
            damped_trend=True,
            seasonal='add',
            seasonal_periods=52
        ).fit()
    else:
        print(f"⚠️ Too little data to simulate with  seasonal model ETS ({len(series)} points). disabling accessibility.")
        model = ExponentialSmoothing(
            series,
            trend='add',
            damped_trend=True,
            seasonal=None
        ).fit()

    forecast_values = model.forecast(steps)
    last_date = series.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=steps, freq='W')
    return pd.DataFrame({'date': future_dates, 'rate': forecast_values})

def forecast_sarima(df_hist: pd.DataFrame, steps: int = 3) -> pd.DataFrame:
    """
    predicts using SARIMA's method.

    Args:
        df_hist (pd.DataFrame): histirical mothods
        steps (int): numb of steps

    Returns:
        pd.DataFrame: Forecast (future only)
    """
    # Prepare times row
    series = df_hist.set_index('date')['rate'].sort_index()
    
    if len(series) < 20:
        raise ValueError("Too little data(min ~20pt. ) to simulate by SARIMA")

    # SARIMA parameters: (p,d,q) x (P,D,Q,s)
    # s=52 — weeks data
    order = (1, 1, 1)      # An unsteady process with a trend
    # Threshold: at least two full seasonsа (52 * 2 = 104 weeks)
    MIN_SEASONAL_LENGTH = 104

    if len(series) >= MIN_SEASONAL_LENGTH:
        seasonal_order = (1, 1, 1, 52)  # Seasonality with a period of 52 weeks
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
    else:
        model = SARIMAX(series, order=order)
        print(f"⚠️ There is not enough data for seasonal SARIMA ({len(series)} weeks). use common model.")


    try:
        fitted_model = model.fit(disp=False)  # disp=False — do not output logs during training
        forecast_values = fitted_model.forecast(steps)

        # generate date
        #last_date = series.index[-1]
        last_date = series.dropna().index[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=steps, freq='W')

        return pd.DataFrame({'date': future_dates, 'rate': forecast_values})

    except Exception as e:
        print(f"❌ Error wile studies of SARIMA: {e}")
        last_value = series.dropna().iloc[-1] # a .dropna for full NaN value
        future_dates = pd.date_range(series.dropna().index[-1] + pd.Timedelta(weeks=1), periods=steps, freq='W')
        return pd.DataFrame({'date': future_dates, 'rate': [last_value] * steps})


def plot_forecasts(
    df_hist: pd.DataFrame,
    forecasts: dict,
    colors: dict = None,
    output_path: str = 'results/forecast_comparison.png'
):
    """
     Builds a continuous graph of history (gray shades) and some forecasts
    Args:
        df_hist: historical data
        forecasts: voc: {'name_model': DataFrame with 'date', 'rate'}
        colors: voc of colors by model
        output_path: save path
    """
    if colors is None:
        colors = {
            'ETS': 'green',
            'SARIMA': 'blue',
            'Prophet': 'purple',
            'XGBoost': 'red'
        }

    plt.figure(figsize=(16, 7))

    # --- 1. History ---
    colors_history = {2023: '#999999', 2024: '#555555', 2025: '#000000'}
    for i in range(len(df_hist) - 1):
        x0, y0 = df_hist.iloc[i]['date'], df_hist.iloc[i]['rate']
        x1, y1 = df_hist.iloc[i+1]['date'], df_hist.iloc[i+1]['rate']
        year = df_hist.iloc[i]['year']
        plt.plot([x0, x1], [y0, y1], color=colors_history[year], linewidth=2, alpha=0.8)

    # Points of history
    for year in [2023, 2024, 2025]:
        data_year = df_hist[df_hist['year'] == year]
        plt.scatter(data_year['date'], data_year['rate'],
                    s=20, zorder=5, color=colors_history[year], edgecolor='white', linewidth=0.5)

    # --- 2. Forecast ---
    main_models = ['ETS', 'SARIMA', 'Prophet', 'XGBoost']
    for name, forecast_df in forecasts.items():
        color = colors.get(name.split(' (')[0], 'blue')
        linestyle = '--' if 'Backtest' not in name else '-.'
        linewidth = 3 if 'Backtest' not in name else 2
        alpha = 0.9 if 'Backtest' not in name else 0.7

        if 'ETS' in name:
            marker = '.'
            markersize = 10
        elif 'SARIMA' in name:
            marker = 'D'
            markersize = 6
        else:
            marker = 'o' 
            markersize = 6
        
        label = name if name in main_models else None
        
        plt.plot(forecast_df['date'], forecast_df['rate'],
                 color=color, linestyle='--', linewidth=3, label=name)
        plt.scatter(forecast_df['date'], forecast_df['rate'],
                    color=color, s=markersize*10, zorder=10, marker=marker, edgecolors='white', linewidth=0.5)

    # --- Design ---
    plt.title('Dynamics of influenza incidence in the Russian Federation - with a forecast for 3 weeks', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Caases / 10 000')
    plt.xticks(rotation=45)
    plt.legend(title='Model', loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.figtext(0.5, 0.01
        ,"If you're surprised by the sharp drop on January 1, it's the number of medical visits, not the number of cases!"
        ,ha="center", fontsize=9, style="italic", alpha=0.7, wrap=True
    )
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

def main():

    DATA_DIR = 'data'
    RESULTS_DIR = 'results'
    OUTPUT_PLOT = os.path.join(RESULTS_DIR, 'forecast_comparison.png')

    try:

        os.makedirs(RESULTS_DIR, exist_ok=True)
        print("🔍 Load data...")
        df = load_data(DATA_DIR)

        print("📊 General forecast for future")
        pred_ets_now = forecast_ets(df, steps=3)
        pred_sarima_now = forecast_sarima(df, steps=3) 

        # Backtesting: forecast in The PAST
        cutoff_2024 = pd.Timestamp('2024-10-01')
        train_2024 = df[df['date'] < cutoff_2024]
        pred_ets_2024 = forecast_ets(train_2024, steps=3)
        pred_sarima_2024 = forecast_sarima(train_2024, steps=3)    
        
        # voc of all forecasts
        forecasts = {
            'ETS': pred_ets_now
            ,'SARIMA': pred_sarima_now
            ,'SARIMA (Backtest 2024)': pred_sarima_2024
            ,'ETS (Backtest 2024)': pred_ets_2024            
        }

        print("📈 Show Forecast:")
        plot_forecasts(df, forecasts, output_path=OUTPUT_PLOT)

        print("✅ Analysis completed successfully!")
    except Exception as e:
        print(f"❌ Errors: {e}")
        raise


if __name__ == '__main__':
    main()