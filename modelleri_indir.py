"""
Yüz tanıma için gerekli model dosyalarını indir
"""
import urllib.request
import os

def dosya_indir(url, dosya_adi):
    if os.path.exists(dosya_adi):
        print(f"{dosya_adi} zaten var")
        return True
    
    print(f"İndiriliyor: {dosya_adi}")
    try:
        urllib.request.urlretrieve(url, dosya_adi)
        print("Tamamlandı!")
        return True
    except Exception as e:
        print(f"Hata: {e}")
        return False

def main():
    print("="*60)
    print("MODEL DOSYALARI İNDİRİLİYOR")
    print("="*60)
    
    os.makedirs("models", exist_ok=True)
    
    dosyalar = [
        {
            "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "dosya": "models/deploy.prototxt"
        },
        {
            "url": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            "dosya": "models/res10_300x300_ssd_iter_140000.caffemodel"
        }
    ]
    
    basarili = 0
    for item in dosyalar:
        if dosya_indir(item["url"], item["dosya"]):
            basarili += 1
    
    print("\n" + "="*60)
    if basarili == len(dosyalar):
        print("Tüm modeller hazır!")
        print("\nŞimdi çalıştırın: python yuz_el_takip.py")
    else:
        print("Bazı dosyalar indirilemedi")
    print("="*60)

if __name__ == "__main__":
    main()

