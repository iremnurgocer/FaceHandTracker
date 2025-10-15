import cv2
import numpy as np
from collections import deque

# -----------------------------
# Yardımcı fonksiyonlar
# -----------------------------
def iqr_range(arr, scale=1.5):
    """Verilen 1D array için IQR tabanlı alt/üst sınır döndürür."""
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    low = q1 - scale * iqr
    high = q3 + scale * iqr
    return low, high

def clamp_range(low, high, vmin, vmax):
    return max(vmin, low), min(vmax, high)

def extract_cheek_samples(hsv_img, face_rect, sample_ratio=0.18):
    """Yüz kutusu içinden yanak bölgelerinden HSV örnekleri çıkarır."""
    x, y, w, h = face_rect
    # Yanağı kabaca yüzün orta-alt kısmından alalım
    cx = x + w // 2
    cy = y + int(h * 0.6)

    sw = int(w * sample_ratio)
    sh = int(h * sample_ratio)

    # Sol ve sağ yanak pencereleri
    left_rect = (x + int(w*0.15) - sw//2, cy - sh//2, sw, sh)
    right_rect = (x + int(w*0.85) - sw//2, cy - sh//2, sw, sh)

    samples = []
    for (rx, ry, rw, rh) in (left_rect, right_rect):
        rx = max(0, rx); ry = max(0, ry)
        rx2 = min(hsv_img.shape[1], rx + rw)
        ry2 = min(hsv_img.shape[0], ry + rh)
        roi = hsv_img[ry:ry2, rx:rx2]
        if roi.size > 0:
            samples.append(roi.reshape(-1, 3))

    if samples:
        return np.concatenate(samples, axis=0)
    return None

def build_skin_range_from_face(hsv_img, face_rect):
    """Yüzden alınan örneklerle adaptif HSV cilt aralığı hesaplar."""
    samples = extract_cheek_samples(hsv_img, face_rect)
    if samples is None or len(samples) < 50:
        # Yeterli örnek yoksa makul bir fallback aralığı
        # (bu aralığı gerekirse ayarlayabilirsiniz)
        return (0, 15), (30, 180), (60, 255)

    H = samples[:, 0].astype(np.float32)
    S = samples[:, 1].astype(np.float32)
    V = samples[:, 2].astype(np.float32)

    h_low, h_high = iqr_range(H, 1.5)
    s_low, s_high = iqr_range(S, 1.5)
    v_low, v_high = iqr_range(V, 1.5)

    h_low, h_high = clamp_range(h_low, h_high, 0, 179)
    s_low, s_high = clamp_range(s_low, s_high, 10, 255)   # S çok düşükse noise artar; min 10
    v_low, v_high = clamp_range(v_low, v_high, 30, 255)   # V çok düşükse gölgeler; min 30

    # Aralığı biraz genişletelim (tolerans)
    h_pad = 8
    s_pad = 40
    v_pad = 40
    h_low = max(0, h_low - h_pad); h_high = min(179, h_high + h_pad)
    s_low = max(0, s_low - s_pad); s_high = min(255, s_high + s_pad)
    v_low = max(0, v_low - v_pad); v_high = min(255, v_high + v_pad)

    return (int(h_low), int(h_high)), (int(s_low), int(s_high)), (int(v_low), int(v_high))

def clean_mask(mask):
    """Maske üzerinde gürültü temizliği."""
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def find_largest_contours(mask, max_count=3, min_area=2500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            big.append((area, (x, y, w, h), cnt))
    big.sort(key=lambda z: z[0], reverse=True)
    return big[:max_count]

def is_hand_like(cnt):
    """Basit morfolojik kontrol: şeklin 'pürüzlülüğü' ve dışbükeyliği."""
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False
    solidity = float(area) / hull_area  # ~0.5-0.95 arası ellerde makul olabilir
    # Çok düz/katı şekilleri ele: solidity aşırı yüksekse (örn. >0.98) reddet
    # Çok dağınık ise (<0.4) reddet
    if 0.45 <= solidity <= 0.98:
        return True
    return False

def centroid_of(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)

# -----------------------------
# Ana uygulama
# -----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı.")
        return

    # Yüz dedektörü
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("Haar cascade yüklenemedi.")
        return

    # Stabilizasyon için son N karedeki tespitler
    history_len = 5
    right_hist = deque(maxlen=history_len)
    left_hist = deque(maxlen=history_len)

    # HSV cilt aralığı cache
    hsv_range = None
    frames_since_face = 0

    print("Başlatılıyor... Çıkış: 'q'")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Aynamış görüntü
        frame = cv2.flip(frame, 1)
        vis = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1) Yüz tespiti
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        face_rect = None
        if len(faces) > 0:
            # En büyük yüz
            face_rect = max(faces, key=lambda r: r[2] * r[3])
            x, y, w, h = face_rect
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 128, 0), 2)
            cv2.putText(vis, "Yuz", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

            # 2) Adaptif cilt araligi (belirli araliklarla/guncel yuz varsa)
            if frames_since_face % 5 == 0:  # her 5 karede bir güncelle
                hsv_range = build_skin_range_from_face(hsv, face_rect)
            frames_since_face = 0
        else:
            frames_since_face += 1

        # 3) Maske
        mask = None
        if hsv_range is not None:
            (hL, hH), (sL, sH), (vL, vH) = hsv_range
            mask = cv2.inRange(hsv, (hL, sL, vL), (hH, sH, vH))
            mask = clean_mask(mask)
        else:
            # Henüz yüz bulunamadıysa bir şey yapmayalım
            cv2.putText(vis, "Yuz bekleniyor...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Hand-Face Labeler", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        # 4) El aday konturları
        candidates = find_largest_contours(mask, max_count=4, min_area=2500)
        right_found = False
        left_found = False

        if face_rect is not None and candidates:
            fx, fy, fw, fh = face_rect
            face_cx = fx + fw // 2
            face_cy = fy + fh // 2

            for area, rect, cnt in candidates:
                if not is_hand_like(cnt):
                    continue

                x, y, w, h = rect
                cx, cy = centroid_of(rect)

                # "Kaldirilmis" kriteri: yüz merkezinin çok altında değilse
                raised = cy < (face_cy + int(1.5 * fh))

                # Yüz kutusunun çok içinde kalan büyük cilt bölgesini (yüzün kendisi) ele
                if (x > fx - 10) and (x + w < fx + fw + 10) and (y > fy - 10) and (y + h < fy + fh + 10):
                    continue

                # Sağ/sol tayini
                if cx > face_cx and raised:
                    label = "Sag El"
                    color = (0, 200, 0)
                    right_found = True
                elif cx < face_cx and raised:
                    label = "Sol El"
                    color = (0, 200, 0)
                    left_found = True
                else:
                    # Şimdilik ilgi alanımız değil
                    continue

                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        right_hist.append(1 if right_found else 0)
        left_hist.append(1 if left_found else 0)

        # Stabilize edilmiş gösterge
        if sum(right_hist) >= max(2, history_len // 2):
            cv2.putText(vis, "Sag El: YUKARIDA", (10, vis.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        if sum(left_hist) >= max(2, history_len // 2):
            cv2.putText(vis, "Sol El: YUKARIDA", (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        # Bilgi panosu
        cv2.putText(vis, "q: cikis", (vis.shape[1] - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Maske penceresini de ufak göstermek isterseniz:
        small_mask = cv2.resize(mask, (0,0), fx=0.4, fy=0.4)
        small_mask_bgr = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
        h, w = small_mask_bgr.shape[:2]
        vis[10:10+h, 10:10+w] = small_mask_bgr

        cv2.imshow("Hand-Face Labeler", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
