import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np


# ==================================================
# DATA MAHASISWA
# ==================================================
NAMA = "Ade Teguh Prakoso"
NIM = "24041013"


# ==================================================
# APLIKASI
# ==================================================
class AplikasiPCD:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Project UTS Pengolahan Citra Digital - {NAMA} | {NIM}")
        self.root.geometry("1350x760")
        self.root.configure(bg="#edf4ec")
        self.root.resizable(False, False)

        self.gambar_asli = None
        self.gambar_hasil = None

        self.lebar_box = 430
        self.tinggi_box = 320

        self.buat_tampilan()

    # ==================================================
    # TAMPILAN GUI
    # ==================================================
    def buat_tampilan(self):
        header = tk.Frame(self.root, bg="#dfe9db", height=90)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="PROJECT UTS PENGOLAHAN CITRA DIGITAL",
            font=("Segoe UI", 24, "bold"),
            bg="#dfe9db",
            fg="#1b1b1b"
        ).pack(pady=(12, 0))

        tk.Label(
            header,
            text=f"{NAMA} | {NIM}",
            font=("Segoe UI", 13),
            bg="#dfe9db",
            fg="#333333"
        ).pack()

        body = tk.Frame(self.root, bg="#edf4ec")
        body.pack(fill="both", expand=True, padx=15, pady=15)

        # ================= MENU KIRI =================
        menu = tk.Frame(body, bg="white", width=260, bd=1, relief="solid")
        menu.pack(side="left", fill="y", padx=(0, 15))
        menu.pack_propagate(False)

        tk.Label(
            menu,
            text="MENU FITUR",
            font=("Segoe UI", 16, "bold"),
            bg="#2f7d32",
            fg="white",
            pady=12
        ).pack(fill="x", padx=10, pady=10)

        self.tombol(menu, "Load Citra", self.load_citra)
        self.tombol(menu, "RGB ke Grayscale", self.grayscale)
        self.tombol(menu, "RGB ke HSV", self.hsv)
        self.tombol(menu, "Brightness", self.brightness)
        self.tombol(menu, "Contrast", self.contrast)
        self.tombol(menu, "Histogram Equalization", self.histogram_equalization)
        self.tombol(menu, "Gamma Correction", self.gamma_correction)
        self.tombol(menu, "Rotasi 90°", self.rotasi)
        self.tombol(menu, "Flip Horizontal", self.flip_horizontal)
        self.tombol(menu, "Edge Detection", self.edge_detection)
        self.tombol(menu, "Simpan Hasil", self.simpan_hasil, "#2f7d32", "white")

        # ================= AREA GAMBAR =================
        area = tk.Frame(body, bg="#edf4ec")
        area.pack(side="right", fill="both", expand=True)

        panel = tk.Frame(area, bg="#edf4ec")
        panel.pack(pady=15)

        self.frame_asli, self.label_asli = self.buat_panel_gambar(panel, "CITRA ASLI", 0)
        self.frame_hasil, self.label_hasil = self.buat_panel_gambar(panel, "CITRA HASIL", 1)

        footer = tk.Label(
            self.root,
            text=f"© 2026 Project Pengolahan Citra Digital - {NAMA} ({NIM})",
            font=("Segoe UI", 10),
            bg="#edf4ec",
            fg="#555555"
        )
        footer.pack(side="bottom", pady=5)

    def buat_panel_gambar(self, parent, judul, kolom):
        frame = tk.Frame(
            parent,
            bg="white",
            width=480,
            height=430,
            bd=2,
            relief="solid"
        )
        frame.grid(row=0, column=kolom, padx=18)
        frame.grid_propagate(False)

        tk.Label(
            frame,
            text=judul,
            font=("Segoe UI", 16, "bold"),
            bg="white",
            fg="#2f7d32"
        ).pack(pady=10)

        label = tk.Label(
            frame,
            bg="#f5f5f5",
            width=self.lebar_box,
            height=self.tinggi_box
        )
        label.pack(padx=15, pady=10)

        return frame, label

    def tombol(self, parent, teks, command, bg="white", fg="black"):
        tk.Button(
            parent,
            text=teks,
            command=command,
            font=("Segoe UI", 10),
            width=24,
            height=1,
            bg=bg,
            fg=fg,
            activebackground="#d7ead3",
            relief="solid",
            bd=1,
            cursor="hand2"
        ).pack(padx=15, pady=5)

    # ==================================================
    # VALIDASI
    # ==================================================
    def cek_gambar(self):
        if self.gambar_asli is None:
            messagebox.showwarning("Peringatan", "Silakan load citra terlebih dahulu!")
            return False
        return True

    # ==================================================
    # TAMPILKAN GAMBAR TANPA TERPOTONG
    # ==================================================
    def tampilkan_gambar(self, img_bgr, label):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)

        image.thumbnail((self.lebar_box, self.tinggi_box))

        canvas = Image.new(
            "RGB",
            (self.lebar_box, self.tinggi_box),
            "#f5f5f5"
        )

        x = (self.lebar_box - image.width) // 2
        y = (self.tinggi_box - image.height) // 2

        canvas.paste(image, (x, y))

        photo = ImageTk.PhotoImage(canvas)
        label.config(image=photo)
        label.image = photo

    # ==================================================
    # FITUR WAJIB
    # ==================================================
    def load_citra(self):
        path = filedialog.askopenfilename(
            title="Pilih Citra",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp")
            ]
        )

        if path:
            gambar = cv2.imread(path)

            if gambar is None:
                messagebox.showerror("Error", "Citra gagal dibuka!")
                return

            self.gambar_asli = gambar
            self.gambar_hasil = gambar.copy()

            self.tampilkan_gambar(self.gambar_asli, self.label_asli)
            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)

    def grayscale(self):
        if self.cek_gambar():
            gray = cv2.cvtColor(self.gambar_asli, cv2.COLOR_BGR2GRAY)
            self.gambar_hasil = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)

    def hsv(self):
        if self.cek_gambar():
            hsv = cv2.cvtColor(self.gambar_asli, cv2.COLOR_BGR2HSV)
            self.gambar_hasil = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)

    def brightness(self):
        if self.cek_gambar():
            self.gambar_hasil = cv2.convertScaleAbs(
                self.gambar_asli,
                alpha=1,
                beta=50
            )

            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)

    def contrast(self):
        if self.cek_gambar():
            self.gambar_hasil = cv2.convertScaleAbs(
                self.gambar_asli,
                alpha=1.6,
                beta=0
            )

            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)

    def histogram_equalization(self):
        if self.cek_gambar():
            ycrcb = cv2.cvtColor(self.gambar_asli, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            self.gambar_hasil = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)

    def gamma_correction(self):
        if self.cek_gambar():
            gamma = 1.8
            inv_gamma = 1.0 / gamma

            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in range(256)
            ]).astype("uint8")

            self.gambar_hasil = cv2.LUT(self.gambar_asli, table)

            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)

    def rotasi(self):
        if self.cek_gambar():
            self.gambar_hasil = cv2.rotate(
                self.gambar_asli,
                cv2.ROTATE_90_CLOCKWISE
            )

            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)

    def flip_horizontal(self):
        if self.cek_gambar():
            self.gambar_hasil = cv2.flip(self.gambar_asli, 1)

            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)

    def edge_detection(self):
        if self.cek_gambar():
            gray = cv2.cvtColor(self.gambar_asli, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(gray, 100, 200)
            self.gambar_hasil = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)

    def simpan_hasil(self):
        if self.gambar_hasil is None:
            messagebox.showwarning("Peringatan", "Belum ada citra hasil yang disimpan!")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPG", "*.jpg"),
                ("BMP", "*.bmp")
            ]
        )

        if path:
            cv2.imwrite(path, self.gambar_hasil)
            messagebox.showinfo("Berhasil", "Citra hasil berhasil disimpan!")


# ==================================================
# RUN PROGRAM
# ==================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AplikasiPCD(root)
    root.mainloop()