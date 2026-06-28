# Tesla Stock Price Prediction DL

This is a Deep Learning and EDA project for predicting Tesla closing prices with SimpleRNN and LSTM models across 1-day, 5-day, and 10-day forecast horizons.

## Project Structure

```text
Tesla-Stock-Price-Prediction-DL/
|
+-- data/
|   +-- TSLA.csv
|
+-- notebooks/
|   +-- Tesla_Stock_Prediction_End_To_End.ipynb
|
+-- outputs/
|   +-- visualizations/
|   +-- predictions/
|   +-- metrics/
|
+-- requirements.txt
+-- README.md
```

## Notebook Coverage

- Data cleaning
- Exploratory data analysis
- Time-series preprocessing
- Feature engineering
- SimpleRNN modeling
- LSTM modeling
- GridSearchCV hyperparameter tuning
- Model comparison and evaluation

## Streamlit Deployment

The Streamlit deployment file is available as `app.py`.

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Modeling Notes

The project uses the `Close` column as the target because the problem statement asks for analysis on closing price. Missing values are handled with time-aware forward fill after sorting by date, with a final backward fill only for possible leading missing values. Train/test splitting is chronological to avoid leakage.