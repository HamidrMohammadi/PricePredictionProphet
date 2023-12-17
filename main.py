import pandas as pd
from fbprophet import Prophet

# Load the dataset
data = pd.read_csv('gold_prices.csv')

# Convert the date column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Rename the columns to 'ds' and 'y' as required by Prophet
data = data.rename(columns={'date': 'ds', 'price': 'y'})

# Create and fit the model
model = Prophet()
model.fit(data)

# Ask for prediction year
prediction_year = input("Enter the year for which you want to make a prediction: ")

# Make future dataframe for the specified year
future = model.make_future_dataframe(periods=365, freq='D', include_history=False)
future['ds'] = pd.to_datetime(prediction_year + '-01-01') + pd.to_timedelta(future.index, unit='D')

# Make predictions
forecast = model.predict(future)

# Print the predicted prices for the specified year
print(forecast[['ds', 'yhat']].tail(10))
