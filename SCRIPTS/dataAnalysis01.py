import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def plot_sector_line_trends(sector_csv_path: str,
                            lookback_months=6,
                            output_path="OUTPUTS/sector_line_6month.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(sector_csv_path)
    df['date'] = pd.to_datetime(df['date'])
    if "indeed_job_postings_index" in df.columns:
        df.rename(columns={"indeed_job_postings_index": "job_postings"}, inplace=True)

    sector_col = "display_name"
    if sector_col not in df.columns:
        print(f"Could not find '{sector_col}' in {sector_csv_path}.")
        return

    sector_avgs = df.groupby(sector_col)['job_postings'].mean().sort_values(ascending=False)
    top_5_sectors = sector_avgs.head(5).index.tolist()

    df_top5 = df[df[sector_col].isin(top_5_sectors)].copy()
    df_top5.set_index('date', inplace=True)

    monthly_top5 = (
        df_top5.groupby(sector_col)['job_postings']
        .apply(lambda x: x.resample('M').mean())
        .reset_index()
    )
    pivoted = monthly_top5.pivot(index='date', columns=sector_col, values='job_postings').fillna(0)

    most_recent = pivoted.index.max()
    if pd.isnull(most_recent):
        print("No valid date data found for the sector CSV")
        return
    cutoff_date = most_recent - pd.DateOffset(months=lookback_months)
    pivoted = pivoted.loc[pivoted.index >= cutoff_date]

    plt.figure(figsize=(10, 6))
    for col in pivoted.columns:
        plt.plot(pivoted.index, pivoted[col], marker='o', label=col)

    plt.title(f"Top 5 Sectors (Last {lookback_months} Months) - Monthly Mean Job Postings")
    plt.xlabel("Date")
    plt.ylabel("Job Postings (Avg Index)")
    plt.legend(title="Sector", loc='upper left')
    plt.tight_layout()

    plt.savefig(output_path)
    print(f"Sector line chart saved to: {output_path}")
    plt.close()


def plot_us_map(state_csv_path: str,
                output_path="OUTPUTS/us_map_states.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(state_csv_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if "indeed_job_postings_index" in df.columns:
        df.rename(columns={"indeed_job_postings_index": "job_postings"}, inplace=True)

    if "State" not in df.columns:
        if "state" in df.columns:
            df.rename(columns={"state": "State"}, inplace=True)
        else:
            print("Could not find 'State' or 'state' in the dataset.")
            return

    state_means = df.groupby("State")['job_postings'].mean().reset_index()
    state_means.rename(columns={"job_postings": "AveragePostings"}, inplace=True)

    state_means["State"] = state_means["State"].str.upper()
    fig = px.choropleth(
        state_means,
        locations="State",
        locationmode="USA-states",
        color="AveragePostings",
        hover_name="State",
        color_continuous_scale="Blues",
        scope="usa",
        labels={"AveragePostings": "Avg Postings"}
    )
    fig.update_layout(
        title_text="Average Job Postings by State (Indeed Index)",
        margin={"r":0,"t":50,"l":0,"b":0}
    )

    fig.write_image(output_path)
    print(f"US map saved to: {output_path}")


def main():
    sector_csv = "DATA/job_postings_by_sector_US.csv"
    state_csv = "DATA/state_job_postings_us.csv"
    plot_sector_line_trends(
        sector_csv_path=sector_csv,
        lookback_months=6,
        output_path="OUTPUTS/sector_line_6month.png"
    )

    plot_us_map(
        state_csv_path=state_csv,
        output_path="OUTPUTS/us_map_states.png"
    )

if __name__ == "__main__":
    main()
