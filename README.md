# Airline Food Demand Prediction

This project is aimed at predicting the total food demand on airline flights based on various features such as flight duration, passenger count, flight hour, business class ratio, and past consumption rates. 

## Dataset Generation
The data is synthetically generated using factors that influence airline food consumption. The dataset (`ucak_yemek_veri_seti.csv`) includes features like:
- Flight Duration
- Passenger Count (Adults & Children)
- Business Class Ratio
- International vs. Domestic
- Flight Hour
- Past Consumption Rate

## Analysis
Exploratory Data Analysis (EDA) is performed, producing:
- Correlation heatmaps.
- Distribution plots for target variables and key features.
- Scatter plots to explore the relationship between features and the target variable.

## Setup
Install the necessary dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

Run the `airline_food_demand_prediction.ipynb` notebook to see the data generation and analysis steps.
