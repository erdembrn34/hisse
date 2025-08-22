# Gerekli kütüphaneler
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

# 1. Veri çekme
symbol = 'AAPL'
data = yf.download(symbol, start='2018-01-01', end='2023-12-31', auto_adjust=True)

# 2. Özellikleri seçme
features = data[['Open','High','Low','Close','Volume']].values

# 3. Veri ölçekleme
scaler = MinMaxScaler(feature_range=(0,1))
scaled_features = scaler.fit_transform(features)

# 4. Eğitim ve test setine ayırma
train_size = int(len(scaled_features) * 0.8)
train_data = scaled_features[:train_size]
test_data = scaled_features[train_size-90:]  # overlap için 90 gün ekliyoruz

# 5. Zaman serisi verisi oluşturma fonksiyonu
def create_dataset(dataset, time_step=90):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i])
        y.append(dataset[i, 3])  # Close fiyat
    return np.array(X), np.array(y)

time_step = 90
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 6. LSTM modeli oluşturma
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 7. Modeli eğitme
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# 8. Tahmin yapma
predicted_train = model.predict(X_train)
predicted_test = model.predict(X_test)

# 9. Ters ölçekleme (Close fiyat)
dummy_train = np.zeros((predicted_train.shape[0], features.shape[1]))
dummy_train[:,3] = predicted_train[:,0]
predicted_train_prices = scaler.inverse_transform(dummy_train)[:,3]

dummy_test = np.zeros((predicted_test.shape[0], features.shape[1]))
dummy_test[:,3] = predicted_test[:,0]
predicted_test_prices = scaler.inverse_transform(dummy_test)[:,3]

# Orijinal Close fiyatları
dummy_y_train = np.zeros((y_train.shape[0], features.shape[1]))
dummy_y_train[:,3] = y_train
original_train = scaler.inverse_transform(dummy_y_train)[:,3]

dummy_y_test = np.zeros((y_test.shape[0], features.shape[1]))
dummy_y_test[:,3] = y_test
original_test = scaler.inverse_transform(dummy_y_test)[:,3]

# 10. RMSE hesaplama
rmse_train = math.sqrt(mean_squared_error(original_train, predicted_train_prices))
rmse_test = math.sqrt(mean_squared_error(original_test, predicted_test_prices))
print(f'RMSE (Train): {rmse_train:.2f}')
print(f'RMSE (Test): {rmse_test:.2f}')

# 11. Grafikle görselleştirme
plt.figure(figsize=(12,6))
plt.plot(range(len(original_train)), original_train, label='Gerçek Train Fiyat')
plt.plot(range(len(predicted_train_prices)), predicted_train_prices, label='Tahmin Train Fiyat')
plt.plot(range(len(original_train), len(original_train)+len(original_test)), original_test, label='Gerçek Test Fiyat')
plt.plot(range(len(original_train), len(original_train)+len(predicted_test_prices)), predicted_test_prices, label='Tahmin Test Fiyat')
plt.title(f'{symbol} Hisse Senedi Fiyat Tahmini - Geleceğe Yönelik')
plt.xlabel('Gün')
plt.ylabel('Fiyat')
plt.legend()
plt.show()
# 12. Modeli kaydetme
model.save('hisse_lstm_model.h5')

import numpy as np
import matplotlib.pyplot as plt

# 1. Son 90 günü al
last_90_days = scaled_features[-90:]  # tüm veri setinden son 90 gün

# 2. 30 gün tahmin için boş liste
future_predictions = []

current_input = last_90_days.copy()

for i in range(30):
    # Model için input shape: [samples, time_step, features]
    X_input = current_input.reshape(1, current_input.shape[0], current_input.shape[1])
    
    # Tahmin yap
    pred_scaled = model.predict(X_input)[0,0]
    
    # Ters ölçekleme
    dummy = np.zeros((1, features.shape[1]))
    dummy[0,3] = pred_scaled  # Close fiyatı
    pred_price = scaler.inverse_transform(dummy)[0,3]
    
    # Tahmini kaydet
    future_predictions.append(pred_price)
    
    # Yeni girdi oluştur: bir gün kaydır, tahmini ekle
    new_row = current_input[-1].copy()
    new_row[3] = pred_scaled  # Close fiyatını tahmin ile güncelle
    current_input = np.vstack([current_input[1:], new_row])

# 3. Sonuçları görselleştirme
plt.figure(figsize=(12,6))
plt.plot(range(len(data)), data['Close'].values, label='Gerçek Fiyat')
plt.plot(range(len(data), len(data)+30), future_predictions, label='Tahmin Edilen Gelecek 30 Gün')
plt.title('AAPL Hisse Senedi - Gelecek 30 Gün Tahmini')
plt.xlabel('Gün')
plt.ylabel('Fiyat')
plt.legend()
plt.show()
