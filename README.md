# Yüz ve El Takip Sistemi

## Gereksinimler

- Webcam ile yüz tanıma
- Sağ el ve sol el işaretleme
- Python 3.9+
- Mediapipe kullanılmamıştır

## Kurulum

```bash
pip install opencv-python numpy
```

## Çalıştırma

```bash
python yuz_el_takip.py
```

veya `BASLAT.bat` dosyasına çift tıkla

## Kullanım

1. Kamera açılır
2. Yüz otomatik işaretlenir
3. İlk 2 saniye elleri hareket ettirin
4. Sağ el → "Sağ El" kutucuğu
5. Sol el → "Sol El" kutucuğu
6. Çıkış: `q`

## Özellikler

- DNN ile yüz tespiti
- Birden fazla yüz desteği
- Sadece eller algılanır (kol değil)
- Cilt + hareket + morfoloji
