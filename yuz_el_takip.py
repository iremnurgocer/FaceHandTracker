"""
Webcam ile yüz ve el takip sistemi
Mediapipe kullanılmamıştır
"""
import cv2
import numpy as np
from collections import deque
import os

class YuzElTakip:
    def __init__(self):
        self.models_ready = False
        
        self.face_proto = "models/deploy.prototxt"
        self.face_model = "models/res10_300x300_ssd_iter_140000.caffemodel"
        
        self.load_models()
        
        self.sag_gecmis = deque(maxlen=3)
        self.sol_gecmis = deque(maxlen=3)
        
        self.cilt_araligi = None
        
        self.arka_plan = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=30, detectShadows=False
        )
        
    def load_models(self):
        if os.path.exists(self.face_proto) and os.path.exists(self.face_model):
            try:
                self.face_net = cv2.dnn.readNetFromCaffe(self.face_proto, self.face_model)
                self.models_ready = True
                print("Modeller yüklendi")
            except Exception as e:
                print(f"Hata: {e}")
        else:
            print("Model dosyaları bulunamadı!")
    
    def yuz_bul(self, frame):
        """Tüm yüzleri bul"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        yuzler = []
        
        for i in range(detections.shape[2]):
            guven = detections[0, 0, i, 2]
            
            if guven > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if x2 > x and y2 > y:
                    yuzler.append((x, y, x2 - x, y2 - y))
        
        return yuzler
    
    def cilt_rengi_ogren(self, hsv, yuz_kutu):
        """Yüzden cilt rengi çıkar"""
        x, y, w, h = yuz_kutu
        
        orta_y = y + int(h * 0.5)
        ornek_boy = int(min(w, h) * 0.12)
        
        ornekler = []
        for px in [0.3, 0.5, 0.7]:
            ornek = hsv[
                orta_y - ornek_boy:orta_y + ornek_boy,
                x + int(w * px) - ornek_boy:x + int(w * px) + ornek_boy
            ]
            if ornek.size > 0:
                ornekler.append(ornek.reshape(-1, 3))
        
        if not ornekler:
            return None
        
        tum_ornekler = np.vstack(ornekler)
        
        h_vals = tum_ornekler[:, 0]
        s_vals = tum_ornekler[:, 1]
        v_vals = tum_ornekler[:, 2]
        
        h_alt, h_ust = np.percentile(h_vals, [20, 80])
        s_alt, s_ust = np.percentile(s_vals, [20, 80])
        v_alt, v_ust = np.percentile(v_vals, [20, 80])
        
        h_alt = max(0, h_alt - 15)
        h_ust = min(179, h_ust + 15)
        s_alt = max(20, s_alt - 40)
        s_ust = min(255, s_ust + 50)
        v_alt = max(40, v_alt - 50)
        v_ust = min(255, v_ust + 50)
        
        return (int(h_alt), int(h_ust)), (int(s_alt), int(s_ust)), (int(v_alt), int(v_ust))
    
    def el_maskesi_olustur(self, frame, hsv, yuzler):
        """El maskesi oluştur, yüzleri çıkar"""
        if self.cilt_araligi is None:
            return None
        
        h, w = frame.shape[:2]
        
        # Cilt maskesi
        (h_alt, h_ust), (s_alt, s_ust), (v_alt, v_ust) = self.cilt_araligi
        cilt_maske = cv2.inRange(hsv, (h_alt, s_alt, v_alt), (h_ust, s_ust, v_ust))
        
        # Hareket maskesi
        hareket_maske = self.arka_plan.apply(frame)
        _, hareket_maske = cv2.threshold(hareket_maske, 200, 255, cv2.THRESH_BINARY)
        
        # YCrCb maskesi
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb_maske = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        
        # Birleştir
        birlesik = cv2.bitwise_and(cilt_maske, hareket_maske)
        birlesik = cv2.bitwise_and(birlesik, ycrcb_maske)
        
        # Morfolojik temizlik
        kernel = np.ones((11, 11), np.uint8)
        birlesik = cv2.medianBlur(birlesik, 11)
        birlesik = cv2.morphologyEx(birlesik, cv2.MORPH_OPEN, kernel, iterations=2)
        birlesik = cv2.morphologyEx(birlesik, cv2.MORPH_CLOSE, kernel, iterations=3)
        birlesik = cv2.GaussianBlur(birlesik, (7, 7), 0)
        
        # Tüm yüzleri maskeden sil
        if yuzler is not None and len(yuzler) > 0:
            for yuz_kutu in yuzler:
                fx, fy, fw, fh = yuz_kutu
                marj_x = int(fw * 0.5)
                marj_y = int(fh * 0.5)
                
                y1 = max(0, fy - marj_y)
                y2 = min(h, fy + fh + marj_y)
                x1 = max(0, fx - marj_x)
                x2 = min(w, fx + fw + marj_x)
                
                birlesik[y1:y2, x1:x2] = 0
        
        return birlesik
    
    def el_mi_kontrol(self, kontur, yuz_alani):
        """El mi değil mi kontrol et"""
        alan = cv2.contourArea(kontur)
        if alan <= 0:
            return False, 0.0
        
        puanlar = []
        
        # Alan - sadece el boyutu
        min_alan = yuz_alani * 0.18
        max_alan = yuz_alani * 0.85
        puanlar.append(1.0 if min_alan < alan < max_alan else 0.0)
        
        # Şekil
        x, y, w, h = cv2.boundingRect(kontur)
        aspect = float(w) / h if h > 0 else 0
        puanlar.append(1.0 if 0.35 < aspect < 2.0 else 0.0)
        
        # Kolları ele
        if w > 300 or h > 400:
            return False, 0.0
        
        # Solidity
        hull = cv2.convexHull(kontur)
        hull_alan = cv2.contourArea(hull)
        if hull_alan > 0:
            solidity = float(alan) / hull_alan
            puanlar.append(1.0 if 0.45 < solidity < 0.88 else 0.0)
        else:
            return False, 0.0
        
        # Extent
        bbox_alan = w * h
        if bbox_alan > 0:
            extent = float(alan) / bbox_alan
            puanlar.append(1.0 if 0.45 < extent < 0.90 else 0.0)
        else:
            puanlar.append(0.0)
        
        # Compactness
        cevre = cv2.arcLength(kontur, True)
        if cevre > 0:
            compactness = (cevre * cevre) / (4 * np.pi * alan) if alan > 0 else 999
            puanlar.append(1.0 if compactness < 25 else 0.0)
        else:
            puanlar.append(0.0)
        
        # Circularity
        if cevre > 0:
            circularity = (4 * np.pi * alan) / (cevre * cevre)
            puanlar.append(1.0 if circularity < 0.7 else 0.0)
        else:
            puanlar.append(0.0)
        
        ortalama = np.mean(puanlar)
        return ortalama >= 0.70, ortalama
    
    def eller_bul(self, maske, yuzler, frame_w, frame_h):
        """Elleri bul"""
        if yuzler is None or len(yuzler) == 0:
            return []
        
        en_buyuk_yuz = max(yuzler, key=lambda y: y[2] * y[3])
        fx, fy, fw, fh = en_buyuk_yuz
        yuz_alani = fw * fh
        
        konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        eller = []
        
        for kontur in konturlar:
            alan = cv2.contourArea(kontur)
            
            # Büyük alanları ele
            if alan > yuz_alani * 0.85:
                continue
            
            x, y, w, h = cv2.boundingRect(kontur)
            
            # Kenar kontrolü
            if x <= 5 or y <= 5 or x + w >= frame_w - 5 or y + h >= frame_h - 5:
                continue
            
            # Uzun nesneleri ele
            if w > 300 or h > 400:
                continue
            
            if w > h * 3 or h > w * 3:
                continue
            
            # Doğrulama
            el_mi, puan = self.el_mi_kontrol(kontur, yuz_alani)
            
            if el_mi:
                eller.append({
                    'kutu': (x, y, w, h),
                    'merkez': (x + w // 2, y + h // 2),
                    'alan': alan,
                    'puan': puan
                })
        
        eller.sort(key=lambda e: e['puan'], reverse=True)
        return eller[:2]
    
    def sag_sol_ayir(self, el, yuzler):
        """Hangi el: sağ mı sol mu?"""
        if yuzler is None or len(yuzler) == 0:
            return None
        
        ex, ey = el['merkez']
        
        # En yakın yüzü bul
        en_yakin_yuz = None
        min_mesafe = float('inf')
        
        for yuz in yuzler:
            fx, fy, fw, fh = yuz
            yuz_mx = fx + fw // 2
            yuz_my = fy + fh // 2
            
            mesafe = np.sqrt((ex - yuz_mx)**2 + (ey - yuz_my)**2)
            if mesafe < min_mesafe:
                min_mesafe = mesafe
                en_yakin_yuz = yuz
        
        if en_yakin_yuz is None:
            return None
        
        fx, fy, fw, fh = en_yakin_yuz
        yuz_mx = fx + fw // 2
        
        # Yüzün altındaysa alma
        if ey > fy + fh + int(fh * 0.6):
            return None
        
        # Yüze çok yakınsa alma
        if abs(ex - yuz_mx) < int(fw * 0.2):
            return None
        
        # Sağ/Sol
        if ex > yuz_mx:
            return "sag"
        else:
            return "sol"
    
    def calistir(self):
        """Ana döngü"""
        if not self.models_ready:
            print("Model yüklenemedi!")
            return
        
        kamera = cv2.VideoCapture(0)
        if not kamera.isOpened():
            print("Kamera açılamadı!")
            return
        
        kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        kamera.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*60)
        print("YÜZ VE EL TAKİP SİSTEMİ")
        print("="*60)
        print("İlk 2 saniye ellerinizi hareket ettirin")
        print("Çıkış: 'q'")
        print("="*60 + "\n")
        
        frame_sayisi = 0
        
        while True:
            ret, frame = kamera.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Yüzleri bul
            yuzler = self.yuz_bul(frame)
            
            if yuzler is not None and len(yuzler) > 0:
                # Tüm yüzleri çiz
                for yuz_kutu in yuzler:
                    fx, fy, fw, fh = yuz_kutu
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                    cv2.putText(frame, "YUZ", (fx, fy - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Cilt rengi güncelle
                if frame_sayisi % 10 == 0:
                    en_buyuk_yuz = max(yuzler, key=lambda y: y[2] * y[3])
                    self.cilt_araligi = self.cilt_rengi_ogren(hsv, en_buyuk_yuz)
            
            # Elleri bul
            sag_bulundu = False
            sol_bulundu = False
            
            if self.cilt_araligi is not None and yuzler is not None and len(yuzler) > 0:
                el_maskesi = self.el_maskesi_olustur(frame, hsv, yuzler)
                
                if el_maskesi is not None:
                    eller = self.eller_bul(el_maskesi, yuzler, w, h)
                    
                    for el in eller:
                        taraf = self.sag_sol_ayir(el, yuzler)
                        
                        ex, ey, ew, eh = el['kutu']
                        
                        if taraf == "sag" and not sag_bulundu:
                            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                            cv2.putText(frame, "Sag El", (ex, ey - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            sag_bulundu = True
                        
                        elif taraf == "sol" and not sol_bulundu:
                            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                            cv2.putText(frame, "Sol El", (ex, ey - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                            sol_bulundu = True
            
            # Stabilizasyon
            self.sag_gecmis.append(1 if sag_bulundu else 0)
            self.sol_gecmis.append(1 if sol_bulundu else 0)
            
            cv2.putText(frame, "q: Cikis", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Yuz ve El Takip", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_sayisi += 1
        
        kamera.release()
        cv2.destroyAllWindows()
        print("Program sonlandı.")

def main():
    sistem = YuzElTakip()
    sistem.calistir()

if __name__ == "__main__":
    main()
