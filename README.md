### Name:Karnan K
### Reg no:212222230062
### Date: 
# Ex.No: 07                                       AUTO REGRESSIVE MODEL




### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Plot the test data and predictions.
### PROGRAM
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the dataset
df = pd.read_csv('airline_baggage_complaints.csv', index_col='date', parse_dates=['date'])

# Perform Augmented Dickey-Fuller test
result = adfuller(df['complaints'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[0:train_size], df[train_size:len(df)]

# Fit the AR model with 13 lags
model = AutoReg(train['complaints'], lags=13)
model_fit = model.fit()
print('Coefficients:', model_fit.params)

# Plot PACF and ACF
plt.figure(figsize=(10,6))
plot_pacf(train['complaints'], lags=20)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

plt.figure(figsize=(10,6))
plot_acf(train['complaints'], lags=20)
plt.title('Autocorrelation Function (ACF)')
plt.show()
# Make predictions using the AR model
predictions = model_fit.predict(start=len(train), end=len(df)-1, dynamic=False)

# Compare the predictions with the test data
plt.figure(figsize=(10,6))
plt.plot(test['complaints'], label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
```

### OUTPUT:

Augmented Dickey-Fuller test

![Screenshot 2024-10-16 095734](https://github.com/user-attachments/assets/bda933c9-28d6-4ba8-af9a-0954d1731bf0)


PACF - ACF

![Screenshot 2024-10-16 095755](https://github.com/user-attachments/assets/c5b2c0d0-7d92-4272-8d4c-c237d6f013dc)

![Screenshot 2024-10-16 095804](https://github.com/user-attachments/assets/5ea16385-dce6-457e-b111-0aa85cdb313c)

FINIAL PREDICTION

![Screenshot 2024-10-16 095821](https://github.com/user-attachments/assets/d68cfaa2-a1f1-48a8-8207-a82f66e67be6)

### RESULT:
Thus the program implemented successfully based on the auto regression function .
