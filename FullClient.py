# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:56:51 2026

@author: Asus
"""

# -*- coding: utf-8 -*-
#Program UTAMA
import os
import sys
import traceback
import tkinter as tk
import threading
from tkinter import messagebox

from ModulPose import HandPose, ClientPose


labels = ["0", "1", "2", "3", "4", "5"]


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def run_thread(func):
    t = threading.Thread(target=func)
    t.daemon = True
    t.start()


def get_no_kamera():
    return int(var_kamera.get())


def get_model_path():
    return resource_path("model.h5")


def sembunyikan_menu():
    root.iconify()


def tampilkan_menu():
    root.deiconify()
    root.lift()


def create_dataset(label):
    try:
        h = HandPose(cam_index=get_no_kamera())

        sembunyikan_menu()
        root.update()

        h.create_dataset(
            label=str(label),
            max_data=20,
            interval=1.0
        )

        h.close()

        tampilkan_menu()

    except Exception as e:
        traceback.print_exc()
        tampilkan_menu()
        messagebox.showerror("Create Dataset Error", str(e))


def training():
    def jalan():
        try:
            h = HandPose(cam_index=get_no_kamera())

            h.train(
                labels,
                model_path=get_model_path()
            )

            h.close()

            messagebox.showinfo("Info", "Training selesai")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Training Error", str(e))

    run_thread(jalan)


def realtime():
    try:
        model_path = get_model_path()

        if not os.path.exists(model_path):
            messagebox.showerror(
                "Model Error",
                "model.h5 tidak ditemukan:\n" + model_path
            )
            return

        h = HandPose(cam_index=get_no_kamera())

        h.load_model(model_path)

        sembunyikan_menu()

        h.realtime(
            labels,
            conf_min=0.6
        )

        h.close()

        tampilkan_menu()

    except Exception as e:
        traceback.print_exc()
        tampilkan_menu()
        messagebox.showerror("Realtime Error", str(e))


def client_cpp():
    try:
        model_path = get_model_path()

        if not os.path.exists(model_path):
            messagebox.showerror(
                "Model Error",
                "model.h5 tidak ditemukan:\n" + model_path
            )
            return

        sembunyikan_menu()

        app = ClientPose(
            noKamera=get_no_kamera(),
            labels=labels,
            model_path=model_path,
            conf_min=0.6
        )

        app.run()

        tampilkan_menu()

    except Exception as e:
        traceback.print_exc()
        tampilkan_menu()
        messagebox.showerror("Client Error", str(e))


# ==========================================================
# GUI
# ==========================================================
root = tk.Tk()

root.title("MENU HAND POSE")
root.geometry("720x500")
root.resizable(False, False)

var_kamera = tk.IntVar(value=0)


judul = tk.Label(
    root,
    text="MENU HAND POSE",
    font=("Arial", 18, "bold")
)
judul.pack(pady=15)


frame_kamera = tk.LabelFrame(
    root,
    text="Pilih Kamera",
    font=("Arial", 11, "bold"),
    padx=15,
    pady=10
)
frame_kamera.pack(pady=5)

tk.Label(
    frame_kamera,
    text="No Kamera:",
    font=("Arial", 11)
).grid(row=0, column=0, padx=5)

tk.Spinbox(
    frame_kamera,
    from_=0,
    to=10,
    width=6,
    textvariable=var_kamera,
    font=("Arial", 11)
).grid(row=0, column=1, padx=5)

tk.Label(
    frame_kamera,
    text="Default = 0",
    font=("Arial", 10)
).grid(row=0, column=2, padx=10)


frame = tk.Frame(root)
frame.pack(pady=10)


kiri = tk.LabelFrame(
    frame,
    text="Create Dataset",
    font=("Arial", 11, "bold"),
    padx=15,
    pady=15
)
kiri.grid(row=0, column=0, padx=20)

for i in range(6):
    tk.Button(
        kiri,
        text=f"{i+1}. Create Dataset {i}",
        width=28,
        height=2,
        command=lambda x=i: create_dataset(x)
    ).pack(pady=4)


kanan = tk.LabelFrame(
    frame,
    text="Proses",
    font=("Arial", 11, "bold"),
    padx=15,
    pady=15
)
kanan.grid(row=0, column=1, padx=20, sticky="n")


tk.Button(
    kanan,
    text="7. Training",
    width=28,
    height=2,
    bg="#d9ead3",
    command=training
).pack(pady=6)

tk.Button(
    kanan,
    text="8. Clasifikasi Realtime",
    width=28,
    height=2,
    bg="#cfe2f3",
    command=realtime
).pack(pady=6)

tk.Button(
    kanan,
    text="9. Run Client",
    width=28,
    height=2,
    bg="#fce5cd",
    command=client_cpp
).pack(pady=6)

tk.Button(
    kanan,
    text="Keluar",
    width=28,
    height=2,
    bg="#f4cccc",
    command=root.destroy
).pack(pady=10)


info = tk.Label(
    root,
    text="Saat OpenCV berjalan, menu disembunyikan. Tekan ESC untuk kembali.",
    font=("Arial", 9)
)
info.pack(pady=10)


root.mainloop()