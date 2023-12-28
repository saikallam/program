# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

temperature = np.random.uniform(10, 35, n_samples)
humidity = np.random.uniform(20, 80, n_samples)
wind_speed = np.random.uniform(1, 10, n_samples)

# Assume a linear relationship with some noise
aqi = 5 * temperature + 3 * humidity + 2 * wind_speed + np.random.normal(0, 5, n_samples)

# Create a DataFrame
data = pd.DataFrame({'Temperature': temperature, 'Humidity': humidity, 'Wind_Speed': wind_speed, 'AQI': aqi})

# Split the data into training and testing sets
X = data[['Temperature', 'Humidity', 'Wind_Speed']]
y = data['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Actual vs. Predicted AQI')
plt.show()