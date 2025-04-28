# CS3-Indeed-Case-Study

This case study focuses on creating forecast predictions for the next five years in the U.S. _Indeed.com_ job posting market, with data sourced from the FRED's "Job Postings on Indeed", whose datasets come from the _Indeed Hiring Lab's_ "Indeed Job Postings Index" GitHub repository [2,4]. Specifically, using the SARIMAX time series forecast model, five-year forecasts will be created regarding _Indeed.com_ daily aggregate job postings across all sectors and daily job postings organized by individual job sectors and U.S. metro cities [7,8,9,10,19,21]. Conclusions drawn from the outputs created with this project's scripts will inform future job seekers and provide helpful information for the job search on _Indeed_.

Below is a sample forecast output that is produced with the `aggregateForecast.py` script for comparison with your outputs. ![image](https://github.com/user-attachments/assets/d81868c2-0eac-41d6-91de-e2c635facb83)

## Software and Packages
*  Python 3.11.4
*  Python Virtual Environment (Linux, Virtual Studio Code, Virtual Environment)
*  Packages: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `seaborn`, `plotly`
  
## Repository Contents
* Hook Document
* `DATA`
  * `aggregate_job_postings_US (2).csv`
  * `job_postings_by_sector_US (3).csv`
  * `metro_job_postings_us (1).csv`
  * `state_job_postings_us (1).csv`
* `SCRIPTS`
  * `dataAnalysis.py`
  * `dataAnalysis01.py`
  * `aggregateForecast.py`
  * `metroForecast.py`
  * `sectorForecast.py`
* `MATERIALS`
  * `CS3 Articles for Further Reading.pdf`
  * ... (rubric)

## How to Reproduce Project Outputs 
1. Download the four datasets from the `DATA` folder in this repository.
2. Follow along and run the `dataAnalysis.py` and `dataAnalysis01.py` scripts to produce some primary EDA analyses.
3. Follow along and run `aggregateForecast.py` to produce a five-year SARIMAX forecast for the aggregate U.S. job postings.
4. Follow along and run `metroForecast.py` to produce a five-year SARIMAX forecast for the top ten U.S. metro cities with the highest number of job postings.
5. Follow along and run `sectorForecast.py` to produce a five-year SARIMAX forecast for the top ten job sectors with the highest number of job postings.
