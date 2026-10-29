# Melbourne Temperature Forecasting

## Project Overview
This project uses time-series forecasting to predict daily minimum temperatures in Melbourne, Australia. It uses historical temperature data from 1981 to 1990 and the Prophet forecasting model to generate future predictions.

The project also includes a Streamlit web application to display historical data, forecast temperatures, and download forecast results.

## Dataset
- Dataset: Daily Minimum Temperatures in Melbourne
- **Period:** 1981–1990
- **Frequency:** Daily
- **Main columns:** Date and Temperature
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/paulbrabban/daily-minimum-temperatures-in-melbourne/data)

## Technologies Used
- Python
- Pandas
- Matplotlib
- Statsmodels
- Prophet
- Streamlit
- Pickle

## Project Workflow
1. Load and explore the dataset.
2. Convert dates into datetime format.
3. Clean invalid temperature values and handle missing values.
4. Analyze temperature trends and seasonality.
5. Split the data chronologically into training and testing sets.
6. Train and evaluate forecasting models.
7. Tune the Prophet model and perform time-series cross-validation.
8. Save the trained model and deploy it using Streamlit.

## Model Evaluation
The tuned Prophet model achieved the following results on the test dataset:

 Metric & Result 

 MAE = 2.031 
 RMSE= 2.572 

Cross-validation was also performed to evaluate model performance across different time periods.

## Project Structure


minimum temperature
- app.py
- daily-minimum-temperature-in-me-csv
- prophet_model.pkl
- README.md
- temperature.ipynb



## Setup Instructions

### 1. Install Python
Install Python or Anaconda on your computer.

### 2. Open the project folder
Open the folder in VS Code and open a terminal.

### 3. Install required libraries


python -m pip install streamlit pandas matplotlib prophet statsmodels


If python is not recognized in your Windows terminal, use the Python interpreter path configured on your computer.

### 4. Run the application

Make sure app.py, prophet_model.pkl`, and the CSV dataset are in the same folder.


python -m streamlit run app.py


The application should open in your web browser.

## Using the Application
1. View historical temperature statistics and the historical temperature graph.
2. Select a forecast duration: 7, 30, 90, or 365 days.
3. View predicted temperatures and the uncertainty interval.
4. Review the forecast table.
5. Download the predictions as a CSV file.

**Note:** The dataset ends on December 31, 1990, so future forecasts begin on January 1, 1991. These are model estimates, not guaranteed temperatures.

## Limitations
- The dataset covers only 1981–1990.
- Predictions beyond the observed period may be uncertain.
- Model performance on historical test data does not guarantee future accuracy.

