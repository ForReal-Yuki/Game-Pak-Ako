# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:32:38 2026

@author: Asus
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:41:10 2026

@author: Asus
"""

# -*- coding: utf-8 -*-
# Program KlasifikasiPose.py

import os
import cv2
import time
import json
import socket
import math
import numpy as np
import mediapipe as mp

from datetime import datetime

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split


# ==========================================================
# BODY POSE
# ==========================================================
class BodyPose:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.TH = 0.5

    def kosong(self):
        return {
            "pa_n": (0, 0), "pa_v": False,
            "pi_n": (0, 0), "pi_v": False,
            "ba_n": (0, 0), "ba_v": False,
            "bi_n": (0, 0), "bi_v": False,
            "hid_n": (0, 0), "hid_v": False,
            "bt": (0.5, 0.5), "bt_v": True,
            "r": 0
        }

    def ekstraksifitur(self, image):
        h, w, _ = image.shape

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if not res.pose_landmarks:
            return self.kosong(), None

        lm = res.pose_landmarks.landmark

        # landmark utama
        hid = (lm[0].x, lm[0].y)      # hidung
        bi  = (lm[11].x, lm[11].y)    # bahu kiri
        ba  = (lm[12].x, lm[12].y)    # bahu kanan

        # gunakan titik tengah tetap
        bt = (0.5, 0.5)

        # pergelangan kiri dan kanan relatif ke 0.5
        pi = (
            (lm[15].x - 0.5) * 1.2 + 0.5,
            (lm[15].y - 0.5) * 1.2 + 0.5
        )

        pa = (
            (lm[16].x - 0.5) * 1.2 + 0.5,
            (lm[16].y - 0.5) * 1.2 + 0.5
        )

        # visibility
        hid_v = lm[0].visibility > self.TH
        bi_v  = lm[11].visibility > self.TH
        ba_v  = lm[12].visibility > self.TH
        pi_v  = lm[15].visibility > self.TH
        pa_v  = lm[16].visibility > self.TH

        bt_v = True
        r = 0

        def norm(p):
            return (p[0], p[1])

        Fit = {
            "pa_n": norm(pa), "pa_v": pa_v,
            "pi_n": norm(pi), "pi_v": pi_v,

            "ba_n": norm(ba), "ba_v": ba_v,
            "bi_n": norm(bi), "bi_v": bi_v,

            "hid_n": norm(hid), "hid_v": hid_v,

            "bt": norm(bt), "bt_v": bt_v,

            "r": r
        }

        print("PA", Fit["pa_n"], Fit["pa_v"])
        print("PI", Fit["pi_n"], Fit["pi_v"])
        print("BT", Fit["bt"], Fit["bt_v"])

        return Fit, lm

    def draw(self, image, lm, Fit):
        if lm is None:
            cv2.putText(
                image,
                "Pose tidak terdeteksi",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            return image

        h, w, _ = image.shape

        titik = {
            0: "hid",
            11: "bi",
            12: "ba",
            15: "pi",
            16: "pa"
        }

        for idx, nama in titik.items():
            x = int(lm[idx].x * w)
            y = int(lm[idx].y * h)

            cv2.circle(image, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                image,
                nama,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # gambar titik tengah 0.5
        bt_x = int(Fit["bt"][0] * w)
        bt_y = int(Fit["bt"][1] * h)

        cv2.circle(image, (bt_x, bt_y), 8, (0, 0, 255), -1)
        cv2.putText(
            image,
            "bt",
            (bt_x + 8, bt_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        return image

    def close(self):
        self.pose.close()
# ==========================================================
# HAND POSE
# ==========================================================
class HandPose:
    def __init__(self, cam_index=0):
        self.cam_index = cam_index

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.model = None

    def distance(self, p1, p2):
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def angle(self, p1, p2):
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    def save_image(self, filename, image):
        cv2.imwrite(filename, image)

    def load_jpg_list(self, dirname):
        if not os.path.exists(dirname):
            return []
        return [f for f in os.listdir(dirname) if f.lower().endswith(".jpg")]

    def extract_features(self, hand_landmarks, w, h, k):
        pts = []

        for lm in hand_landmarks.landmark:
            pts.append([lm.x * w, lm.y * h])

        pts = np.array(pts, dtype=np.float32)

        l1 = self.distance(pts[0], pts[5])
        l2 = self.distance(pts[0], pts[4])
        l3 = self.distance(pts[0], pts[8])
        l4 = self.distance(pts[0], pts[12])
        l5 = self.distance(pts[0], pts[16])
        l6 = self.distance(pts[0], pts[20])
        l7 = self.distance(pts[4], pts[8])
        l8 = self.distance(pts[8], pts[12])
        l9 = self.distance(pts[12], pts[16])
        l10 = self.distance(pts[16], pts[20])

        if k == 1:
            ki, ka = 1, 0
        else:
            ki, ka = 0, 1

        dx = 1 + self.angle(pts[0], pts[5]) / np.pi

        ref = l1 if l1 > 1e-6 else 1e-6

        features = np.array([
            l1 / ref,
            l2 / ref,
            (l3 / ref) - 1,
            (l4 / ref) - 1,
            (l5 / ref) - 1,
            (l6 / ref) - 1,
            l7 / ref,
            l8 / ref,
            l9 / ref,
            l10 / ref,
            ki,
            ka,
            dx
        ], dtype=np.float32)

        return features, pts

    def extract_from_frame(self, frame):
        frameres = frame.copy()
        h, w, _ = frameres.shape

        rgb = cv2.cvtColor(frameres, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        lf = []

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label

                if handedness == "Left":
                    k = 1
                else:
                    k = 2

                # Aktifkan drawing untuk melihat garis mapping
                self.mp_draw.draw_landmarks(
                    frameres,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                features, pts = self.extract_features(hand_landmarks, w, h, k)

                lf.append([handedness, features, pts])

        return {
            "Fit": lf,
            "Frame": frameres
        }

    def create_dataset1(self, label, max_data=20, interval=1.0):
        os.makedirs(label, exist_ok=True)
        counter = len(self.load_jpg_list(label))

        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
        last_save = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            hasil = self.extract_from_frame(frame)
            lf = hasil["Fit"]
            frameres = hasil["Frame"]

            now = time.time()

            if len(lf) > 0 and (now - last_save) >= interval:
                name = datetime.now().strftime("%y%m%d%H%M%S%f")[:-3] + ".jpg"
                path = os.path.join(label, name)

                self.save_image(path, frame)

                counter += 1
                last_save = now

                print(f"{counter}/{max_data}")

            cv2.imshow("Dataset", frameres)

            if counter >= max_data or cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
    def create_dataset3(self, label, max_data=20, interval=1.0):
        os.makedirs(label, exist_ok=True)
        counter = -5
    
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
        last_save = 0
    
        while True:
            ret, frame = cap.read()
    
            if not ret:
                break
    
            frame = cv2.flip(frame, 1)
    
            hasil = self.extract_from_frame(frame)
            lf = hasil["Fit"]
            frameres = hasil["Frame"]
    
            now = time.time()
    
            # ===============================
            # TAMPILKAN LABEL DAN COUNTER
            # ===============================
            cv2.putText(
                frameres,
                f"Label : {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )
    
            if counter < 1:
                warna_counter = (0, 0, 255)   # merah
            else:
                warna_counter = (0, 255, 0)   # hijau
    
            cv2.putText(
                frameres,
                f"Counter : {counter}/{max_data}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                warna_counter,
                2
            )
    
            if len(lf) > 0 and (now - last_save) >= interval:
                counter += 1
                last_save = now
    
                if counter >= 1:
                    name = datetime.now().strftime("%y%m%d%H%M%S%f")[:-3] + ".jpg"
                    path = os.path.join(label, name)
    
                    self.save_image(path, frame)
    
                    print(f"{counter}/{max_data}")
                else:
                    print(f"{counter}/{max_data} - belum disimpan")
    
            cv2.imshow("Dataset", frameres)
    
            key = cv2.waitKey(1) & 0xFF
    
            if counter > max_data or key == 27 or key == 13:
                break
    
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    
    def create_dataset(self, label, max_data=20, interval=1.0):
        os.makedirs(label, exist_ok=True)
        counter = -5
    
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
        last_save = 0
    
        while True:
            ret, frame = cap.read()
    
            if not ret:
                break
    
            frame = cv2.flip(frame, 1)
    
            hasil = self.extract_from_frame(frame)
            lf = hasil["Fit"]
            frameres = hasil["Frame"]
    
            now = time.time()
    
            # ===============================
            # TAMPILKAN LABEL DAN COUNTER
            # ===============================
            cv2.putText(
                frameres,
                f"Label : {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )
    
            if counter < 1:
                warna_counter = (0, 0, 255)   # merah
            else:
                warna_counter = (0, 255, 0)   # hijau
    
            cv2.putText(
                frameres,
                f"Counter : {counter}/{max_data}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                warna_counter,
                2
            )
    
            if (now - last_save) >= interval:
                counter += 1
                last_save = now
    
                if counter >= 1 and len(lf) > 0:
                    name = datetime.now().strftime("%y%m%d%H%M%S%f")[:-3] + ".jpg"
                    path = os.path.join(label, name)
    
                    self.save_image(path, frame)
    
                    print(f"{counter}/{max_data}")
                else:
                    print(f"{counter}/{max_data} - belum disimpan")
    
            cv2.imshow("Dataset", frameres)
    
            key = cv2.waitKey(1) & 0xFF
    
            if counter > max_data or key == 27 or key == 13:
                break
    
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
    

    def load_dataset(self, labels):
        lx, ly = [], []

        n = len(labels)
        Kelas = np.eye(n, dtype=np.float32)

        for i, lb in enumerate(labels):
            files = self.load_jpg_list(lb)

            for f in files:
                img = cv2.imread(os.path.join(lb, f))

                if img is None:
                    continue

                hasil = self.extract_from_frame(img)
                lf = hasil["Fit"]

                for item in lf:
                    lx.append(item[1])
                    ly.append(Kelas[i])

        return np.array(lx), np.array(ly)

    def train(self, labels, model_path="model.h5"):
        X, y = self.load_dataset(labels)

        if len(X) == 0:
            print("Dataset kosong")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=1
        )

        self.model = Sequential([
            Input(shape=(X.shape[1],)),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(len(labels), activation="softmax")
        ])

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=16
        )

        loss, acc = self.model.evaluate(X_test, y_test)

        print("Loss:", loss)
        print("Accuracy:", acc)

        self.model.save(model_path)

    def load_model(self, path):
        self.model = load_model(path)

    def realtime(self, labels, conf_min=0.6):
        if self.model is None:
            print("Model belum diload")
            return

        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)

        prev_time = time.time()

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            hasil = self.extract_from_frame(frame)
            lf = hasil["Fit"]
            frameres = hasil["Frame"]

            fps = 1.0 / max(time.time() - prev_time, 1e-6)
            prev_time = time.time()

            for item in lf:
                handedness = item[0]
                features = item[1]
                pts = item[2]

                y_pred = self.model.predict(
                    features.reshape(1, -1),
                    verbose=0
                )[0]

                idx = np.argmax(y_pred)
                conf = y_pred[idx]

                if conf >= conf_min:
                    label = labels[idx]
                else:
                    label = "Unknown"

                x = int(np.min(pts[:, 0]))
                y = int(np.min(pts[:, 1])) - 10

                cv2.putText(
                    frameres,
                    f"{handedness}: {label} ({conf:.2f})",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.putText(
                frameres,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.imshow("Realtime", frameres)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def close(self):
        self.hands.close()

# ==========================================================
# CLIENT PYTHON KE C++ SERVER
# ==========================================================
class ClientPose:
    def __init__(
        self,
        noKamera=0,
        labels=None,
        model_path="model.h5",
        conf_min=0.6,
        host="127.0.0.1",
        port=5005
    ):
        if labels is None:
            labels = ["0", "1", "2", "3", "4", "5"]

        self.noKamera = noKamera
        self.labels = labels
        self.model_path = model_path
        self.conf_min = conf_min

        self.host = host
        self.port = port

        self.sock = None
        self.last_try_connect = 0
        self.last_send = 0

        self.body = BodyPose()

        self.hand = HandPose(cam_index=noKamera)
        self.hand.load_model(model_path)

        self.cap = None

    def connect_cpp(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(0.3)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(None)

            print("Connect ke C++")

        except:
            self.sock = None

    def kirim(self, data):
        if self.sock is None:
            return

        try:
            pesan = json.dumps(data) + "\n"
            self.sock.sendall(pesan.encode("utf-8"))

        except:
            try:
                self.sock.close()
            except:
                pass

            self.sock = None

    def buka_camera(self):
        self.cap = cv2.VideoCapture(self.noKamera, cv2.CAP_V4L2)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return self.cap.isOpened()

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)

        # =====================================
        # BODY hanya untuk gambar/draw
        # =====================================
        # =====================================
        # BODY dimatikan sementara agar tidak lag 
        # (karena game hanya butuh tangan)
        # =====================================
        # FitBody, lmBody = self.body.ekstraksifitur(frame)

        # frame = self.body.draw(
        #     frame,
        #     lmBody,
        #     FitBody
        # )

        # =====================================
        # HAND
        # =====================================
        hasilHand = self.hand.extract_from_frame(frame)

        FitHand = hasilHand["Fit"]
        frame = hasilHand["Frame"]

        # =====================================
        # DEFAULT DATA KIRI DAN KANAN
        # =====================================
        kiri_x = 0.000
        kiri_y = 0.000
        kiri_label = -1
        kiri_v = 0

        kanan_x = 0.000
        kanan_y = 0.000
        kanan_label = -1
        kanan_v = 0

        tinggi, lebar, _ = frame.shape

        # =====================================
        # PROSES SETIAP TANGAN
        # =====================================
        for item in FitHand:
            handedness = item[0]
            features = item[1]
            pts = item[2]

            y_pred = self.hand.model.predict(
                features.reshape(1, -1),
                verbose=0
            )[0]

            idx = np.argmax(y_pred)
            conf = float(y_pred[idx])

            if conf > self.conf_min:
                label = int(self.labels[idx])
            else:
                label = -1

            # koordinat pergelangan tangan landmark 0
            x_norm = round(float(pts[0][0]) / lebar, 3)
            y_norm = round(float(pts[0][1]) / tinggi, 3)

            if handedness == "Left":
                kiri_x = x_norm
                kiri_y = y_norm
                kiri_label = label
                kiri_v = 1

            elif handedness == "Right":
                kanan_x = x_norm
                kanan_y = y_norm
                kanan_label = label
                kanan_v = 1

            # tampilkan label di frame
            x_text = int(np.min(pts[:, 0]))
            y_text = int(np.min(pts[:, 1])) - 10

            cv2.putText(
                frame,
                f"{handedness}: {label} ({conf:.2f})",
                (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # =====================================
        # DATA RINGKAS UNTUK SERVER C++
        # =====================================
        Fitur = {
            "Kx": kiri_x,
            "Ky": kiri_y,
            "Kl": kiri_label,
            "Kv": kiri_v,

            "Rx": kanan_x,
            "Ry": kanan_y,
            "Rl": kanan_label,
            "Rv": kanan_v
        }

        return frame, Fitur

    def run(self):
        if not self.buka_camera():
            print("Camera gagal dibuka")
            return

        prev_time = time.time()

        while True:
            # reconnect ke C++ tiap 1 detik jika belum connect
            if self.sock is None:
                now = time.time()

                if now - self.last_try_connect >= 1.0:
                    self.connect_cpp()
                    self.last_try_connect = now

            ret, frame = self.cap.read()

            if not ret:
                break

            frame, Fitur = self.process_frame(frame)

            # kirim ke C++ tiap 0.1 detik
            now_send = time.time()

            if self.sock is not None and now_send - self.last_send >= 0.1:
                self.kirim(Fitur)
                self.last_send = now_send

            fps = 1.0 / max(
                time.time() - prev_time,
                1e-6
            )

            prev_time = time.time()

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Kiri: {Fitur['Kl']} V:{Fitur['Kv']}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Kanan: {Fitur['Rl']} V:{Fitur['Rv']}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            if self.sock is not None:
                teks = "C++ Connected"
                warna = (0, 255, 0)
            else:
                teks = "C++ Not Connected"
                warna = (0, 0, 255)

            cv2.putText(
                frame,
                teks,
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                warna,
                2
            )

            cv2.imshow("Client Pose", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.close()

    def close(self):
        if self.cap is not None:
            self.cap.release()

        if self.sock is not None:
            self.sock.close()

        cv2.destroyAllWindows()
        cv2.waitKey(1)
    def close(self):
        if self.cap is not None:
            self.cap.release()

        if self.sock is not None:
            self.sock.close()

        self.body.close()
        self.hand.close()

        cv2.destroyAllWindows()


# ==========================================================
# TEST LANGSUNG TANPA MENU
# ==========================================================
if __name__ == "__main__":
    labels = ["0", "1", "2", "3", "4", "5"]

    app = ClientPose(
        noKamera=0,
        labels=labels,
        model_path="model.h5",
        conf_min=0.6,
        host="127.0.0.1",
        port=5005
    )

    app.run()