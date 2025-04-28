## Input: metro_job_postings_us.csv
## Output: top_sector_forecasts.png


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def load_data(csv_path: str) -> pd.DataFrame:
    """
    1. Reads the CSV with columns:
         date, metro, cbsa_code, indeed_job_postings_index
    2. Renames 'indeed_job_postings_index' -> 'job_postings'.
    3. Extracts the part after the comma in 'metro' (e.g., "Abilene, TX" -> "TX" 
       or "Huntington-Ashland, WV-KY-OH" -> "WV-KY-OH").
    4. Splits multi-state combos like "WV-KY-OH" into separate rows 
       (using df.explode). Each row now has exactly one state code.
    5. Renames that code -> 'sector' for consistency with the forecast code.
    """
    df = pd.read_csv(csv_path)

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Rename 'indeed_job_postings_index' to 'job_postings'
    if "indeed_job_postings_index" in df.columns:
        df.rename(columns={"indeed_job_postings_index": "job_postings"}, inplace=True)
    
    # Ensure we have a 'metro' column
    if "metro" not in df.columns:
        raise ValueError("CSV must contain a 'metro' column with entries like 'City, ST' or 'City, ST1-ST2'.")

    # 1) Extract the portion after the comma, e.g. "Abilene, TX" -> "TX" 
    #    or "Some Metro, WV-KY-OH" -> "WV-KY-OH"
    df['states_str'] = df['metro'].str.split(",").str[-1].str.strip()

    # 2) Split multi-state combos on the dash, e.g. "WV-KY-OH" -> ["WV","KY","OH"]
    #    Then .explode() them into separate rows.
    df['states_list'] = df['states_str'].apply(lambda x: x.split('-'))
    df = df.explode('states_list')

    # 3) Clean up the whitespace around each state code
    df['states_list'] = df['states_list'].str.strip()

    # 4) Rename 'states_list' -> 'sector' so we can forecast by 'sector'
    df.rename(columns={'states_list': 'sector'}, inplace=True)

    # Drop the helper column
    df.drop(columns=['states_str'], inplace=True)

    return df


def sarimax_forecast_by_sector(df: pd.DataFrame, forecast_months=60):
    """
    1) Group data by 'sector' (i.e., the 2-letter state).
    2) Sort & resample monthly.
    3) Fit a SARIMAX model (order=(1,1,1),(1,0,1,12)).
    4) Compute in-sample R².
    5) Forecast the next 'forecast_months' steps.

    Returns a list of dicts, each with:
      'sector': str (state code),
      'r_squared': float,
      'historical': pd.Series (monthly),
      'forecast': pd.Series,
      'forecast_final': float
    """
    forecast_results = []
    all_states = df['sector'].unique()

    for state_code in all_states:
        sub_df = df[df['sector'] == state_code].copy()
        sub_df.sort_values('date', inplace=True)
        sub_df.set_index('date', inplace=True)

        monthly_series = sub_df['job_postings'].resample('ME').mean().dropna()
        # Must have at least 12 data points to fit a basic model
        if len(monthly_series) < 12:
            continue

        monthly_series = monthly_series.asfreq('ME')
        model = sm.tsa.statespace.SARIMAX(
            monthly_series,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)

        # In-sample R²
        fitted_vals = results.fittedvalues.reindex(monthly_series.index)
        sse = np.sum((monthly_series - fitted_vals)**2)
        sst = np.sum((monthly_series - monthly_series.mean())**2)
        r_squared = 1 - sse/sst if sst != 0 else 0

        # Forecast
        forecast_object = results.get_forecast(steps=forecast_months)
        mean_forecast = forecast_object.predicted_mean

        forecast_results.append({
            'sector': state_code,
            'r_squared': r_squared,
            'historical': monthly_series,
            'forecast': mean_forecast,
            'forecast_final': mean_forecast.iloc[-1]
        })

    return forecast_results


def plot_top_sector_forecasts(forecast_results, top_n=10, output_path="OUTPUTS/top_sector_forecasts.png"):
    """
    1) Sort states by their final forecast value (descending).
    2) Plot them all on one shared figure, bridging historical -> forecast.
    3) Include R² in the legend. 
    4) No grouping of multi-state combos, as each state is now separate.
    """
    # Sort descending by the final forecast value
    forecast_results.sort(key=lambda x: x['forecast_final'], reverse=True)
    top_results = forecast_results[:top_n]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(12, 8))

    # We'll cycle through default colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, res in enumerate(top_results):
        color = colors[i % len(colors)]

        # Historical
        hist_index = res['historical'].index
        hist_values = res['historical'].values
        plt.plot(hist_index, hist_values, color=color)

        # Forecast
        fc_index = res['forecast'].index
        fc_values = res['forecast'].values
        plt.plot(fc_index, fc_values, linestyle='--', color=color,
                 label=f"{res['sector']} (R²={res['r_squared']:.2f})")

        # Bridge line between last hist & first forecast point
        if len(hist_index) > 0 and len(fc_index) > 0:
            last_hist_date = hist_index[-1]
            last_hist_val = hist_values[-1]
            first_fc_date = fc_index[0]
            first_fc_val = fc_values[0]
            if first_fc_date > last_hist_date:
                plt.plot([last_hist_date, first_fc_date],
                         [last_hist_val, first_fc_val],
                         color=color, linestyle=':')

    plt.title("Top 10 States Forecast (SARIMAX) - 5 Years Ahead")
    plt.xlabel("Date")
    plt.ylabel("Job Postings (Index)")
    plt.legend(title="State")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_forecast():
    """
    Main driver:
      - Reads 'metro_job_postings_us.csv'.
      - Splits multi-state combos (e.g. "WV-KY-OH") into separate rows.
      - Forecasts each state separately.
      - Plots the top 10 states by final forecast on ONE chart.
      - Ensures states like WV, KY, and OH are NOT grouped into a single line.
    """
    csv_path = "DATA/metro_job_postings_us.csv"
    df = load_data(csv_path)

    # 5-year (60-month) forecast
    forecast_results = sarimax_forecast_by_sector(df, forecast_months=60)

    if forecast_results:
        plot_top_sector_forecasts(forecast_results, top_n=10,
                                  output_path="OUTPUTS/metro_forecasts.png")
        print("Forecast plot saved to OUTPUTS/metro_forecasts.png")
    else:
        print("No states had sufficient data for forecasting.")


if __name__ == "__main__":
    run_forecast()
