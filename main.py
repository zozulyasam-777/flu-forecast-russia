# main.py
"""
Prediction of influenza incidence in Russia.
Author: Sam Zozulya
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


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
    Predicts next 'steps' weeks with ETS.

    Args:
        df_hist (pd.DataFrame): Historical data before forecast point
        steps (int): Number of steps (weeks) for forecast

    Returns:
        pd.DataFrame: Forecast (future only)
    """
    series = df_hist.set_index('date')['rate'].sort_index()

    if len(series) < 10:
        raise ValueError("Too little data to simulate")

    model = ExponentialSmoothing(
        series,
        trend='add',
        damped_trend=True,
        seasonal='add',
        seasonal_periods=52
    ).fit()

    forecast_values = model.forecast(steps)

    last_date = series.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=steps, freq='W')

    return pd.DataFrame({'date': future_dates, 'rate': forecast_values})


def plot_with_forecast(df: pd.DataFrame, forecast: pd.DataFrame, output_path: str):
    """
    Builds a continuous graph of history (gray shades) and forecast (green).
    The color of the segment depends on the year of its starting point.
    """
    plt.figure(figsize=(16, 7))

    # Colors by year
    colors = {
        2023: '#999999',  # light gray
        2024: '#555555',  # gray
        2025: '#000000'   # black
    }

    # --- 1. Historical data ---
    for i in range(len(df) - 1):
        x0, y0 = df.iloc[i]['date'], df.iloc[i]['rate']
        x1, y1 = df.iloc[i+1]['date'], df.iloc[i+1]['rate']
        year = df.iloc[i]['year']
        plt.plot([x0, x1], [y0, y1], color=colors[year], linewidth=2, alpha=0.8)

    # Dots for clarity
    for year in [2023, 2024, 2025]:
        data_year = df[df['year'] == year]
        plt.scatter(data_year['date'], data_year['rate'],
                    s=20, zorder=5, color=colors[year], edgecolor='white', linewidth=0.5)

    # --- 2. Forecast ---
    plt.plot(forecast['date'], forecast['rate'], color='green', linestyle='--', linewidth=3, label='ETS Forecast')
    plt.scatter(forecast['date'], forecast['rate'], color='green', s=50, zorder=10)

    # --- Design ---
    plt.title('Dynamics of influenza incidence in the Russian Federation - with a forecast for 3 weeks', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Cases / 10 000')
    plt.xticks(rotation=45)
    plt.legend(title='data type')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Graph with forecast saved: {output_path}")
    plt.show()


def main():
    """Basic application logic."""
    DATA_DIR = 'data'
    RESULTS_DIR = 'results'
    OUTPUT_PLOT = os.path.join(RESULTS_DIR, 'flu_trend_with_forecast.png')

    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print("🔍 Loading data...")
        df = load_data(DATA_DIR)

        print("📊 Forecast with ETS...")
        forecast = forecast_ets(df, steps=3)

        print("📈 Plotting with forecast...")
        plot_with_forecast(df, forecast, OUTPUT_PLOT)

        # Displaying the Forecast
        print("\n🔮 Forecast for the next 3 weeks:")
        for _, row in forecast.iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['rate']:.2f} cases / 10 th.")

        print("✅ Analysis completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == '__main__':
    main()
