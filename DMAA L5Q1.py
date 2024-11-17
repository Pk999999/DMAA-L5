import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

dataset_path = 'TAMO Historical Data.csv'
price_data = pd.read_csv(dataset_path)

price_data['Date'] = pd.to_datetime(price_data['Date'], format='%d-%m-%Y')
price_data = price_data.sort_values('Date')
price_data['Price'] = pd.to_numeric(price_data['Price'], errors='coerce')
price_data = price_data.dropna(subset=['Price'])
price_data.set_index('Date', inplace=True)

price_data['MA_50'] = price_data['Price'].rolling(window=50).mean()
price_data['MA_200'] = price_data['Price'].rolling(window=200).mean()
price_data['MA_365'] = price_data['Price'].rolling(window=365).mean()
price_data['MA_500'] = price_data['Price'].rolling(window=500).mean()

def apply_arima(series, arima_order):
    arima_model = ARIMA(series.dropna(), order=arima_order)
    return arima_model.fit()

arima_params = (5, 1, 0)

arima_results = {
    'Price (Original)': apply_arima(price_data['Price'], arima_params),
    '50-Day MA': apply_arima(price_data['MA_50'], arima_params),
    '200-Day MA': apply_arima(price_data['MA_200'], arima_params),
    '365-Day MA': apply_arima(price_data['MA_365'], arima_params),
    '500-Day MA': apply_arima(price_data['MA_500'], arima_params),
}

for key, arima_result in arima_results.items():
    print(f"ARIMA Model Summary for {key}:\n")
    print(arima_result.summary())
    print("\n" + "=" * 80 + "\n")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(price_data['Price'].dropna(), ax=axes[0])
axes[0].set_title('Autocorrelation Function - Price (Original)')

plot_pacf(price_data['Price'].dropna(), ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function - Price (Original)')

plt.tight_layout()
plt.show()