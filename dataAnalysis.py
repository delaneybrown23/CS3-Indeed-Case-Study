## Initial Data Analysis of Job Postings by Sector Over Time
## Input File: job_postings_by_sector_US.csv
## Output File: sector_summary.png


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

input_file = pd.read_csv("DATA/job_postings_by_sector_US.csv", parse_dates=["date"])

## Create the analysis graph using plt from matplotlib
plt.figure(figsize=(12, 6))
for sector in input_file['display_name'].unique():
    sector_data = input_file[input_file['display_name'] == sector]
    plt.plot(sector_data['date'], sector_data['indeed_job_postings_index'], label=sector)
plt.xlabel('Date')
plt.ylabel('Indeed Job Postings Index')
plt.title('Job Postings Index by Sector Over Time')
plt.legend()


## place the analysis graph in outputs folder
output_folder = "OUTPUTS"
os.makedirs(output_folder, exist_ok=True) 
output_file = os.path.join(output_folder, "sector_summary.png")
plt.savefig(output_file)