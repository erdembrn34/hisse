# Hisse Senedi Fiyat Tahmini - LSTM

## Proje Amacı
Bu proje, geçmiş hisse senedi fiyat verilerini kullanarak kısa vadeli fiyat hareketlerini tahmin etmeyi amaçlar. LSTM (Long Short-Term Memory) modeli ile zaman serisi analizi yapılarak geleceğe yönelik fiyat tahminleri gerçekleştirilmiştir.

---

## Veri Seti
- Kaynak: Yahoo Finance
- Kapsam: Son 5 yıla ait günlük hisse senedi fiyatları (Open, High, Low, Close, Volume)
- Örnek hisse senedi: AAPL

---

## Yöntem
- **Model:** LSTM (3 katmanlı)
- **Veri Hazırlığı:**  
  - Min-Max normalizasyon  
  - Zaman serisi oluşturma (time_step=90)
- **Eğitim & Test:**  
  - Eğitim: %80  
  - Test: %20  
- **Tahmin:**  
  - Eğitim ve test seti üzerinde tahmin  
  - Son 30 gün için geleceğe yönelik tahmin

---

## Kullanım
1. Gerekli kütüphaneleri yükle:  
```bash
pip install yfinance numpy matplotlib scikit-learn tensorflow
python hisse_lstm.py

Sonuçlar

RMSE (Train): ~3.9
RMSE (Test): ~4.5

Model, geçmiş fiyat trendlerini ve kısa vadeli geleceği tahmin edebilmektedir.
Öğrenilen Dersler
Zaman serisi modellemede LSTM kullanımı
Hiperparametre optimizasyonu ve dropout etkisi
Veri ölçeklemenin tahmin doğruluğu üzerindeki önemi
Gelecek tahminleri için adım adım tahmin yaklaşımı
