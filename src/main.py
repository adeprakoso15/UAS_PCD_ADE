"""
=========================================================================
 UJIAN AKHIR SEMESTER (UAS) - PENGOLAHAN CITRA DIGITAL
 Project : Aplikasi GUI Computer Vision - Praproses, Segmentasi &
           Pengenalan Pola (Tema Dataset B - Jenis Buah)
 Nama    : Ade Teguh Prakoso
 NIM     : 24041013

 Alur pipeline aplikasi:
     Load Citra -> Praproses (7 teknik) -> Segmentasi (Otsu) ->
     Ekstraksi Fitur -> Pengenalan Pola / Klasifikasi Buah

 Kelas buah yang dikenali (4 kelas, sesuai tema B "Jenis Buah"):
     Apel, Jeruk, Pisang, Anggur

 Konsep tampilan: "Sidebar Dashboard" - sidebar navigasi ramping di kiri
 (bisa discroll), panel citra & log hasil di kanan. Dibuat ringkas agar
 muat dalam satu layar laptop standar (1280x720 ke atas).
=========================================================================
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import random

# ==================================================
# DATA MAHASISWA
# ==================================================
NAMA = "Ade Teguh Prakoso"
NIM = "24041013"
DATASET_DIR = "dataset_buah"

# =========================================================
#  PALET WARNA -- KONSEP "SIDEBAR DASHBOARD"
#  (sidebar gelap slate + area kerja terang, aksen emerald)
# =========================================================
SIDEBAR_BG      = "#1c2333"
SIDEBAR_BG_2    = "#20293c"
SIDEBAR_TEXT    = "#cbd3e1"
SIDEBAR_MUTED   = "#7c879c"
SIDEBAR_HEAD    = "#4fd1a5"

APP_BG          = "#eef1f5"
CARD_BG         = "#ffffff"
CARD_BORDER     = "#dde3ea"
IMG_BOX_BG      = "#0f1420"
TEXT_DARK       = "#1f2430"
TEXT_MUTED      = "#6b7280"

ACCENT          = "#2f9e6e"      # tombol biasa (hover)
ACCENT_MAIN     = "#e08a1e"      # tombol menu utama (pengenalan pola)
ACCENT_SAVE     = "#2563eb"      # tombol simpan

FONT_TITLE   = ("Segoe UI Semibold", 15)
FONT_SUB     = ("Segoe UI", 9)
FONT_SECTION = ("Segoe UI Semibold", 9)
FONT_BTN     = ("Segoe UI", 9)
FONT_PANEL   = ("Segoe UI Semibold", 11)
FONT_LOG     = ("Consolas", 9)


# =========================================================
#  PEMBUATAN DATASET CONTOH (SINTETIS) - TEMA JENIS BUAH
#  Apel (bulat, merah) | Jeruk (bulat, oranye) |
#  Pisang (memanjang, kuning) | Anggur (kluster bulat kecil, ungu)
# =========================================================
def _kanvas_putih(ukuran=400):
    # latar sedikit tidak rata (bukan putih polos sempurna) agar tiap
    # gambar tidak identik dan lebih menyerupai foto asli
    dasar = random.randint(248, 255)
    img = np.ones((ukuran, ukuran, 3), dtype="uint8") * dasar
    noise = np.random.randint(-3, 4, (ukuran, ukuran, 3))
    img = np.clip(img.astype(int) + noise, 240, 255).astype("uint8")
    return img


def _acak_warna(warna_dasar, spread=18):
    """Menggeser tiap kanal B,G,R secara independen agar warna tiap
    sampel benar-benar berbeda, bukan cuma satu nilai jitter global."""
    hasil = []
    for c in warna_dasar:
        hasil.append(int(np.clip(c + random.randint(-spread, spread), 0, 255)))
    return tuple(hasil)


def _warna_acak_hsv(rentang_hue, sat_range=(150, 255), val_range=(150, 235)):
    """Mengambil sampel warna acak langsung di ruang HSV, dibatasi pada
    rentang Hue tertentu. Ini menjamin variasi warna antar sampel tetap
    berada di rentang Hue yang benar untuk kelasnya masing-masing (tidak
    akan 'nyelonong' ke rentang Hue kelas lain), sambil tetap membuat
    kecerahan/saturasi tiap sampel berbeda-beda.
    rentang_hue: tuple (h_min, h_max) ATAU list berisi beberapa tuple
    (dipilih salah satu secara acak) - untuk kelas warna yang melingkar
    seperti merah (mendekati 0 maupun 179).
    """
    if isinstance(rentang_hue[0], (tuple, list)):
        h_min, h_max = random.choice(rentang_hue)
    else:
        h_min, h_max = rentang_hue
    h = random.randint(h_min, h_max)
    s = random.randint(*sat_range)
    v = random.randint(*val_range)
    px = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr)


def _tambah_noda(img, cx, cy, r, jumlah_range=(2, 5), warna_gelap_offset=35):
    """Menambahkan beberapa noda/blemish kecil di permukaan buah supaya
    tiap sampel punya tekstur unik (tidak identik satu sama lain)."""
    jumlah = random.randint(*jumlah_range)
    for _ in range(jumlah):
        ang = random.uniform(0, 2 * np.pi)
        rr = random.uniform(0, r * 0.7)
        px = int(cx + rr * np.cos(ang))
        py = int(cy + rr * np.sin(ang))
        rad = random.randint(3, 8)
        warna_asli = img[py, px].astype(int)
        warna_noda = tuple(np.clip(warna_asli - warna_gelap_offset, 0, 255).tolist())
        cv2.circle(img, (px, py), rad, warna_noda, -1)


def _gambar_apel(warna_dasar=None):
    img = _kanvas_putih()
    # Hue merah pada OpenCV melingkar di sekitar 0/179 -> ambil salah satu ujung
    warna = warna_dasar or _warna_acak_hsv([(0, 3), (176, 179)], sat_range=(170, 255), val_range=(150, 230))
    r = random.randint(65, 118)
    cx = 200 + random.randint(-45, 45)
    cy = 200 + random.randint(-35, 35)
    cv2.circle(img, (cx, cy), r, warna, -1)
    # highlight (pantulan cahaya): versi lebih terang dari warna buah itu
    # sendiri (BUKAN putih murni) supaya tidak "menyatu" dengan latar putih
    # saat disegmentasi, dan diletakkan jauh dari tepi agar tidak membuat
    # siluet buah menjadi cekung (menjaga solidity tetap tinggi).
    warna_highlight = tuple(int(c + (255 - c) * 0.55) for c in warna)
    hx = cx + random.randint(-r // 4, r // 8)
    hy = cy - random.randint(int(r * 0.35), int(r * 0.5))
    cv2.circle(img, (hx, hy), max(8, int(r * 0.18)), warna_highlight, -1)
    _tambah_noda(img, cx, cy, r)
    # tangkai & daun, panjang/sudut acak, daun kadang tidak ada
    panjang_tangkai = random.randint(18, 32)
    sudut_tangkai = random.randint(-15, 15)
    ujung_tangkai = (cx + int(panjang_tangkai * np.sin(np.radians(sudut_tangkai))),
                     cy - r - panjang_tangkai)
    cv2.line(img, (cx, cy - r + 5), ujung_tangkai, (60, 90, 40), random.randint(4, 6))
    if random.random() < 0.7:
        cv2.ellipse(img, (cx + random.randint(10, 26), cy - r - panjang_tangkai + 8),
                     (random.randint(13, 19), random.randint(6, 10)),
                     random.randint(-45, -15), 0, 360, (50, 130, 40), -1)
    return img


def _gambar_jeruk(warna_dasar=None):
    img = _kanvas_putih()
    warna = warna_dasar or _warna_acak_hsv((12, 17), sat_range=(180, 255), val_range=(160, 235))
    r = random.randint(65, 118)
    cx = 200 + random.randint(-40, 40)
    cy = 200 + random.randint(-35, 35)
    cv2.circle(img, (cx, cy), r, warna, -1)
    # tekstur kulit jeruk: jumlah & sebaran bintik acak per sampel
    jumlah_bintik = random.randint(25, 55)
    for _ in range(jumlah_bintik):
        ang = random.uniform(0, 2 * np.pi)
        rr = random.uniform(0, r - 8)
        px = int(cx + rr * np.cos(ang))
        py = int(cy + rr * np.sin(ang))
        cv2.circle(img, (px, py), random.randint(1, 2), (0, 100, 200), -1)
    _tambah_noda(img, cx, cy, r, jumlah_range=(1, 3), warna_gelap_offset=25)
    return img


def _gambar_pisang(warna_dasar=None):
    img = _kanvas_putih()
    warna = warna_dasar or _warna_acak_hsv((24, 31), sat_range=(170, 250), val_range=(170, 240))
    pusat = (200 + random.randint(-30, 30), 200 + random.randint(-25, 25))
    sudut = random.randint(-35, 35)
    panjang = random.randint(120, 195)
    lebar = random.randint(32, 62)
    cv2.ellipse(img, pusat, (panjang, lebar), sudut, 0, 360, warna, -1)
    # ujung lebih gelap, kadang ada bintik kecoklatan di sepanjang badan
    ujung = (int(pusat[0] + panjang * 0.85 * np.cos(np.radians(sudut))),
             int(pusat[1] + panjang * 0.85 * np.sin(np.radians(sudut))))
    cv2.circle(img, ujung, random.randint(8, 13), (10, 140, 160), -1)
    for _ in range(random.randint(0, 4)):
        t = random.uniform(-0.6, 0.6)
        px = int(pusat[0] + t * panjang * np.cos(np.radians(sudut)))
        py = int(pusat[1] + t * panjang * np.sin(np.radians(sudut)))
        cv2.circle(img, (px, py), random.randint(2, 4), (10, 160, 170), -1)
    return img


def _gambar_anggur(warna_dasar=None):
    img = _kanvas_putih()
    warna = warna_dasar or _warna_acak_hsv((132, 154), sat_range=(150, 230), val_range=(110, 200))
    cx = 200 + random.randint(-30, 30)
    cy = 170 + random.randint(-20, 20)
    r_biji = random.randint(16, 27)
    pola_lengkap = [(-1, 0), (1, 0), (0, 1), (-2, 2), (0, 2), (2, 2),
                    (-1, 3), (1, 3), (0, 4.6), (-2.2, 3.6), (2.2, 3.6)]
    jumlah_biji = random.randint(7, len(pola_lengkap))
    baris = random.sample(pola_lengkap, jumlah_biji)
    for dx, dy in baris:
        px = int(cx + dx * (r_biji * 0.9))
        py = int(cy + dy * (r_biji * 0.9))
        goyang_x = random.randint(-4, 4)
        goyang_y = random.randint(-3, 3)
        r_ini = r_biji + random.randint(-4, 4)
        warna_biji = _acak_warna(warna, spread=8)
        cv2.circle(img, (px + goyang_x, py + goyang_y), r_ini, warna_biji, -1)
    panjang_tangkai = random.randint(int(r_biji * 1.3), int(r_biji * 2.2))
    cv2.line(img, (cx, cy - int(r_biji * 1.6)),
             (cx + random.randint(-8, 8), cy - int(r_biji * 1.6) - panjang_tangkai),
             (60, 90, 40), random.randint(4, 6))
    cv2.ellipse(img, (cx - 22 + random.randint(-8, 8), cy - int(r_biji * 3.3)),
                 (random.randint(16, 22), random.randint(8, 12)),
                 random.randint(10, 30), 0, 360, (50, 130, 40), -1)
    return img


def setup_dataset_buah(paksa_buat_ulang=False):
    """Membuat folder dataset contoh berisi citra sintetis 4 kelas buah,
    dipakai untuk pengujian jika mahasiswa belum memiliki foto asli."""
    if os.path.exists(DATASET_DIR) and not paksa_buat_ulang:
        return
    os.makedirs(DATASET_DIR, exist_ok=True)
    kelas = {
        "Apel": _gambar_apel,
        "Jeruk": _gambar_jeruk,
        "Pisang": _gambar_pisang,
        "Anggur": _gambar_anggur,
    }
    for nama_kelas, fungsi in kelas.items():
        folder = os.path.join(DATASET_DIR, nama_kelas)
        os.makedirs(folder, exist_ok=True)
        for i in range(1, 8):
            img = fungsi()
            cv2.imwrite(os.path.join(folder, f"{nama_kelas.lower()}_{i}.jpg"), img)


# =========================================================
#  ATURAN KLASIFIKASI (RULE-BASED) - LAPISAN 3
#  Berdasarkan 4 fitur yang diekstraksi di Lapisan 2:
#   F1: Hue dominan warna (modus histogram Hue, 0-179, ruang HSV OpenCV)
#   F2: Luas area objek (px)
#   F3: Aspect ratio (lebar/tinggi dari minAreaRect)
#   F4: Solidity (luas kontur / luas convex hull)
# =========================================================
def klasifikasi_buah(hue, area, aspect_ratio, solidity):
    """Mengembalikan (nama_kelas, warna_bgr) berdasarkan rentang nilai fitur.

    CATATAN PERBAIKAN (dibanding versi awal):
    Versi awal mengecek `solidity < 0.90` di paling depan, SEBELUM Hue sama
    sekali dicek. Di foto sintetis itu tidak masalah, tapi di foto ASLI
    (kulit jeruk berpori, tekstur apel yang tidak sempurna bulat, dsb)
    solidity objek tunggal pun sering turun sedikit di bawah 0.90 hanya
    karena tekstur permukaan -- akibatnya Jeruk/Apel yang jelas-jelas benar
    dari sisi warna malah "dibajak" jadi Anggur duluan. Maka di versi ini
    Hue dicek LEBIH DULU untuk kelas yang rentang warnanya jelas berbeda
    (Jeruk, Pisang), dan solidity hanya dipakai sebagai pembeda saat Hue
    berada di zona "merah" yang memang tumpang-tindih antara Apel dan
    Anggur merah tua/marun (lihat catatan Anggur di bawah).
    """
    # 1) Bentuk memanjang -> Pisang. TAPI: untaian anggur yang difoto
    #    memanjang juga bisa punya aspect ratio cukup tinggi (mis. ~2.0),
    #    jadi aspect ratio SAJA tidak cukup jadi penentu. Digabung dengan
    #    gerbang Hue kuning/kuning-hijau (ciri warna khas pisang) supaya
    #    untaian anggur (merah/ungu) tidak ikut "terjaring" jadi Pisang.
    if aspect_ratio >= 1.8 and 15 <= hue <= 45:
        return "Pisang", (10, 200, 235)
    # Kalau bentuknya SANGAT memanjang (ar >= 3.0), itu ciri geometris yang
    # sudah sangat kuat mengarah ke pisang apa pun pembacaan Hue-nya (pada
    # data anggur asli, aspect ratio tertinggi yang teramati masih di bawah
    # ini), jadi dipakai sebagai penentu cadangan.
    if aspect_ratio >= 3.0:
        return "Pisang", (10, 200, 235)

    # 2) Rentang Hue oranye -> Jeruk (dicek sebelum aturan solidity apa pun,
    #    supaya tekstur kulit jeruk yang membuat solidity sedikit < 0.90
    #    tidak lagi salah "dibajak" jadi Anggur)
    if 8 <= hue <= 20:
        return "Jeruk", (15, 140, 250)

    # 3) Rentang Hue kuning -> Pisang (untuk foto pisang yang bentuknya
    #    tidak cukup memanjang untuk lolos aturan aspect ratio di atas,
    #    mis. tampak atas / bertumpuk)
    if 21 <= hue <= 40:
        return "Pisang", (10, 200, 235)

    # 4) Rentang Hue ungu tua -> Anggur (sesuai dataset sintetis / anggur
    #    ungu asli, bukan yang merah marun)
    if 100 <= hue <= 165:
        return "Anggur", (150, 40, 120)

    # 5) Zona Hue "merah" (dekat 0 maupun dekat 179) -> di sinilah Apel dan
    #    Anggur merah tua/marun SECARA ALAMI tumpang-tindih, karena warna
    #    kulit anggur merah gelap dan kulit apel merah bisa berada di
    #    rentang Hue yang nyaris sama pada citra asli. Hue saja tidak cukup
    #    untuk memisahkan keduanya, sehingga solidity dipakai sebagai
    #    pembeda tambahan: kluster anggur (banyak butir bulat saling
    #    menempel) cenderung punya solidity lebih rendah (kontur berlekuk
    #    di sela-sela butir) dibanding satu buah apel yang solid membulat.
    #    Ini BUKAN pembeda sempurna -- foto anggur close-up satu butir atau
    #    anggur yang terlihat sangat rapat tetap bisa salah terbaca sebagai
    #    Apel. Keterbatasan ini sengaja didokumentasikan di laporan sebagai
    #    kelas yang belum 100% andal dengan pendekatan rule-based Hue+bentuk.
    if hue <= 7 or hue >= 169:
        if solidity < 0.90:
            return "Anggur", (150, 40, 120)
        return "Apel", (40, 40, 200)

    return "Tidak Dikenali", (150, 150, 150)


# =========================================================
#  APLIKASI UTAMA
# =========================================================
class AplikasiPCD:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Studio Pengenalan Buah - PCD | {NAMA} - {NIM}")
        self.root.geometry("1280x700")
        self.root.minsize(1080, 620)
        self.root.configure(bg=APP_BG)

        setup_dataset_buah()

        self.gambar_asli = None
        self.gambar_hasil = None
        self.box_width = 400
        self.box_height = 260
        self.status_var = tk.StringVar(value="Siap digunakan - silakan muat citra buah untuk memulai.")

        self._setup_style()
        self.buat_tampilan()

    # -----------------------------------------------------------
    #  STYLE TTK
    # -----------------------------------------------------------
    def _setup_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Side.TButton", background=SIDEBAR_BG_2, foreground=SIDEBAR_TEXT,
                         font=FONT_BTN, borderwidth=0, focusthickness=0, padding=(10, 7),
                         anchor="w")
        style.map("Side.TButton",
                  background=[("active", "#2a3550")],
                  foreground=[("active", "#ffffff")])

        style.configure("Main.TButton", background=ACCENT_MAIN, foreground="#241300",
                         font=("Segoe UI Semibold", 10), borderwidth=0, padding=(10, 10))
        style.map("Main.TButton", background=[("active", "#f0a13f")])

        style.configure("Save.TButton", background=ACCENT_SAVE, foreground="#ffffff",
                         font=("Segoe UI Semibold", 9), borderwidth=0, padding=(10, 9))
        style.map("Save.TButton", background=[("active", "#3b76ef")])

        style.configure("Vertical.TScrollbar", background=SIDEBAR_BG, troughcolor=SIDEBAR_BG,
                         bordercolor=SIDEBAR_BG, arrowcolor=SIDEBAR_TEXT)

    # -----------------------------------------------------------
    #  TAMPILAN UTAMA
    # -----------------------------------------------------------
    def buat_tampilan(self):
        # ============== TOP BAR (ramping, bukan header besar) ==============
        topbar = tk.Frame(self.root, bg=SIDEBAR_BG, height=54)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="\U0001F34E  Studio Pengenalan Buah", font=FONT_TITLE,
                 bg=SIDEBAR_BG, fg="#ffffff").pack(side="left", padx=18)
        tk.Label(topbar, text=f"UAS Pengolahan Citra Digital  •  {NAMA}  •  NIM {NIM}",
                 font=FONT_SUB, bg=SIDEBAR_BG, fg=SIDEBAR_MUTED).pack(side="right", padx=18)

        # ============== BODY: sidebar kiri (scrollable) + konten kanan ==============
        body = tk.Frame(self.root, bg=APP_BG)
        body.pack(fill="both", expand=True)

        sidebar_outer = tk.Frame(body, bg=SIDEBAR_BG, width=248)
        sidebar_outer.pack(side="left", fill="y")
        sidebar_outer.pack_propagate(False)

        self._buat_sidebar_scrollable(sidebar_outer)

        # ---- Konten kanan ----
        konten = tk.Frame(body, bg=APP_BG)
        konten.pack(side="left", fill="both", expand=True, padx=16, pady=14)

        # baris panel citra
        panel_row = tk.Frame(konten, bg=APP_BG)
        panel_row.pack(fill="both", expand=True)
        panel_row.grid_columnconfigure(0, weight=1)
        panel_row.grid_columnconfigure(1, weight=1)
        panel_row.grid_rowconfigure(0, weight=1)

        self.label_asli = self.panel_gambar(panel_row, "CITRA ASLI", "#e08a1e", 0)
        self.label_hasil = self.panel_gambar(panel_row, "CITRA HASIL", "#2f9e6e", 1)

        # panel log hasil (menggantikan tulisan tebal di atas citra)
        log_card = tk.Frame(konten, bg=CARD_BG, highlightbackground=CARD_BORDER, highlightthickness=1)
        log_card.pack(fill="x", pady=(12, 0))
        log_top = tk.Frame(log_card, bg=CARD_BG)
        log_top.pack(fill="x", padx=14, pady=(8, 2))
        tk.Label(log_top, text="Log Hasil Proses", font=FONT_PANEL, bg=CARD_BG, fg=TEXT_DARK).pack(side="left")
        self.log_text = tk.Text(log_card, height=6, font=FONT_LOG, bg="#f7f9fb", fg=TEXT_DARK,
                                 relief="flat", wrap="word", padx=10, pady=6)
        self.log_text.pack(fill="x", padx=12, pady=(0, 10))
        self.log_text.configure(state="disabled")

        # ============== STATUS BAR ==============
        statusbar = tk.Frame(self.root, bg=SIDEBAR_BG, height=28)
        statusbar.pack(fill="x", side="bottom")
        statusbar.pack_propagate(False)
        tk.Label(statusbar, textvariable=self.status_var, font=("Segoe UI", 8),
                 bg=SIDEBAR_BG, fg=SIDEBAR_TEXT, anchor="w").pack(side="left", padx=14)
        tk.Label(statusbar, text=f"{NAMA} · {NIM}", font=("Segoe UI", 8),
                 bg=SIDEBAR_BG, fg=SIDEBAR_MUTED, anchor="e").pack(side="right", padx=14)

    # -----------------------------------------------------------
    #  SIDEBAR SCROLLABLE (agar semua tombol muat walau layar pendek)
    # -----------------------------------------------------------
    def _buat_sidebar_scrollable(self, parent):
        canvas = tk.Canvas(parent, bg=SIDEBAR_BG, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview, style="Vertical.TScrollbar")
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=SIDEBAR_BG)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(window_id, width=event.width)

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", _on_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _judul_seksi(teks, warna=SIDEBAR_HEAD, pad_top=14):
            tk.Label(inner, text=teks, font=FONT_SECTION, bg=SIDEBAR_BG, fg=warna,
                      anchor="w").pack(fill="x", padx=14, pady=(pad_top, 4))

        def _tombol(teks, command, style="Side.TButton"):
            b = ttk.Button(inner, text=teks, command=command, style=style)
            b.pack(fill="x", padx=12, pady=2)
            return b

        # ---- Lapisan 1: Praproses ----
        _judul_seksi("LAPISAN 1 · PRAPROSES CITRA", pad_top=10)
        _tombol("\u2b06  Load Citra", self.load_citra)
        _tombol("Blur / Smoothing (Gaussian)", self.blur_smoothing)
        _tombol("Sharpening (Unsharp Mask)", self.sharpening)
        _tombol("Edge Detection (Canny)", self.edge_detection)
        _tombol("Denoising (Salt&Pepper + Median)", self.denoising)
        _tombol("Operasi Morfologi (Open+Close)", self.morfologi)
        _tombol("Adaptive Threshold", self.adaptive_threshold)
        _tombol("Histogram Equalization", self.histogram_equalization)

        # ---- Lapisan 2: Segmentasi & Fitur ----
        _judul_seksi("LAPISAN 2 · SEGMENTASI & FITUR")
        _tombol("Segmentasi (Otsu Threshold)", self.segmentasi_otsu)
        _tombol("Ekstraksi Fitur", self.ekstraksi_fitur)
        _tombol("\u21bb  Reset ke Citra Asli", self.reset_citra)

        # ---- Lapisan 3: Pengenalan Pola ----
        _judul_seksi("LAPISAN 3 · MENU UTAMA")
        main_btn = ttk.Button(inner, text="\u25c8  PENGENALAN POLA (KLASIFIKASI BUAH)",
                               command=self.pengenalan_pola, style="Main.TButton")
        main_btn.pack(fill="x", padx=12, pady=(4, 8))

        # ---- Lain-lain ----
        _judul_seksi("LAINNYA")
        _tombol("Buat Dataset Contoh Buah", self.buat_dataset_ui)
        save_btn = ttk.Button(inner, text="\U0001F4be  Simpan Hasil", command=self.simpan_hasil, style="Save.TButton")
        save_btn.pack(fill="x", padx=12, pady=(4, 16))

    # -----------------------------------------------------------
    #  KOMPONEN: PANEL CITRA (kartu sederhana, bukan gaya galeri emas)
    # -----------------------------------------------------------
    def panel_gambar(self, parent, judul, warna, kolom):
        card = tk.Frame(parent, bg=CARD_BG, highlightbackground=CARD_BORDER, highlightthickness=1)
        card.grid(row=0, column=kolom, padx=(0 if kolom == 0 else 8, 8 if kolom == 0 else 0), sticky="nsew")

        top = tk.Frame(card, bg=CARD_BG)
        top.pack(fill="x", padx=14, pady=(10, 6))
        tk.Frame(top, bg=warna, width=4, height=16).pack(side="left", padx=(0, 8))
        tk.Label(top, text=judul, font=FONT_PANEL, bg=CARD_BG, fg=TEXT_DARK).pack(side="left")

        box_wrap = tk.Frame(card, bg=IMG_BOX_BG)
        box_wrap.pack(padx=14, pady=(0, 14))
        label = tk.Label(box_wrap, bg=IMG_BOX_BG, width=self.box_width, height=self.box_height)
        label.pack()
        return label

    # -----------------------------------------------------------
    #  LOG PANEL
    # -----------------------------------------------------------
    def _log(self, teks, bersihkan=False):
        self.log_text.configure(state="normal")
        if bersihkan:
            self.log_text.delete("1.0", "end")
        self.log_text.insert("end", teks + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # -----------------------------------------------------------
    #  VALIDASI & TAMPILKAN GAMBAR
    # -----------------------------------------------------------
    def cek_gambar(self):
        if self.gambar_asli is None:
            messagebox.showwarning("Peringatan", "Silakan load citra buah terlebih dahulu!")
            return False
        return True

    def tampilkan_gambar(self, img, label):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)
        image.thumbnail((self.box_width, self.box_height))
        canvas_img = Image.new("RGB", (self.box_width, self.box_height), IMG_BOX_BG)
        x = (self.box_width - image.width) // 2
        y = (self.box_height - image.height) // 2
        canvas_img.paste(image, (x, y))
        photo = ImageTk.PhotoImage(canvas_img)
        label.config(image=photo)
        label.image = photo

    # -----------------------------------------------------------
    #  LOAD & RESET
    # -----------------------------------------------------------
    def load_citra(self):
        path = filedialog.askopenfilename(
            title="Pilih Citra Buah",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
        gambar = cv2.imread(path)
        if gambar is None:
            messagebox.showerror("Error", "Citra gagal dibuka!")
            return
        self.gambar_asli = gambar
        self.gambar_hasil = gambar.copy()
        self.tampilkan_gambar(self.gambar_asli, self.label_asli)
        self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)
        self.status_var.set(f"Citra dimuat: {os.path.basename(path)}")
        self._log(f">> Citra dimuat: {os.path.basename(path)}", bersihkan=True)

    def reset_citra(self):
        if self.cek_gambar():
            self.gambar_hasil = self.gambar_asli.copy()
            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)
            self.status_var.set("Citra hasil dikembalikan ke citra asli.")
            self._log(">> Citra hasil direset ke citra asli.")

    def buat_dataset_ui(self):
        setup_dataset_buah(paksa_buat_ulang=True)
        messagebox.showinfo(
            "Dataset Dibuat",
            f"Dataset contoh 4 kelas buah (Apel, Jeruk, Pisang, Anggur) berhasil dibuat "
            f"di folder '{DATASET_DIR}/'.\nGunakan tombol 'Load Citra' untuk mengujinya."
        )
        self.status_var.set("Dataset contoh buah berhasil dibuat / diperbarui.")
        self._log(f">> Dataset contoh buah dibuat ulang di folder '{DATASET_DIR}/'.")

    # -----------------------------------------------------------
    #  LAPISAN 1 - PRAPROSES CITRA (7 teknik, minimal 5 diwajibkan)
    # -----------------------------------------------------------
    def blur_smoothing(self):
        if self.cek_gambar():
            self.gambar_hasil = cv2.GaussianBlur(self.gambar_asli, (15, 15), 0)
            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)
            self.status_var.set("Praproses: Gaussian Blur / Smoothing diterapkan.")
            self._log(">> [Praproses] Gaussian Blur / Smoothing diterapkan.")

    def sharpening(self):
        if self.cek_gambar():
            blur = cv2.GaussianBlur(self.gambar_asli, (0, 0), sigmaX=3)
            self.gambar_hasil = cv2.addWeighted(self.gambar_asli, 1.5, blur, -0.5, 0)
            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)
            self.status_var.set("Praproses: Sharpening (Unsharp Masking) diterapkan.")
            self._log(">> [Praproses] Sharpening (Unsharp Masking) diterapkan.")

    def edge_detection(self):
        if self.cek_gambar():
            gray = cv2.cvtColor(self.gambar_asli, cv2.COLOR_BGR2GRAY)
            self.gambar_hasil = cv2.Canny(gray, 80, 180)
            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)
            self.status_var.set("Praproses: Edge Detection (Canny) diterapkan.")
            self._log(">> [Praproses] Edge Detection (Canny) diterapkan.")

    def denoising(self):
        if self.cek_gambar():
            noisy = self.gambar_asli.copy()
            prob = 0.02
            mask = np.random.rand(*noisy.shape[:2])
            noisy[mask < prob / 2] = [0, 0, 0]
            noisy[mask > 1 - prob / 2] = [255, 255, 255]
            self.gambar_hasil = cv2.medianBlur(noisy, 5)
            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)
            self.status_var.set("Praproses: noise Salt & Pepper ditambahkan lalu direstorasi (Median Filter).")
            self._log(">> [Praproses] Noise Salt & Pepper ditambahkan, lalu direstorasi dengan Median Filter.")

    def morfologi(self):
        if self.cek_gambar():
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(self.gambar_asli, cv2.MORPH_OPEN, kernel)
            self.gambar_hasil = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)
            self.status_var.set("Praproses: Operasi Morfologi (Opening + Closing) diterapkan.")
            self._log(">> [Praproses] Operasi Morfologi (Opening + Closing) diterapkan.")

    def adaptive_threshold(self):
        if self.cek_gambar():
            gray = cv2.cvtColor(self.gambar_asli, cv2.COLOR_BGR2GRAY)
            self.gambar_hasil = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)
            self.status_var.set("Praproses: Adaptive Thresholding diterapkan.")
            self._log(">> [Praproses] Adaptive Thresholding diterapkan.")

    def histogram_equalization(self):
        if self.cek_gambar():
            ycrcb = cv2.cvtColor(self.gambar_asli, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            self.gambar_hasil = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)
            self.status_var.set("Praproses: Histogram Equalization diterapkan.")
            self._log(">> [Praproses] Histogram Equalization diterapkan.")

    # -----------------------------------------------------------
    #  LAPISAN 2 - SEGMENTASI (Otsu Thresholding)
    # -----------------------------------------------------------
    def _segmentasi_mask(self, sumber):
        gray = cv2.cvtColor(sumber, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def segmentasi_otsu(self):
        if self.cek_gambar():
            mask = self._segmentasi_mask(self.gambar_asli)
            self.gambar_hasil = mask
            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)
            self.status_var.set("Segmentasi: Otsu Thresholding (objek buah vs latar) diterapkan.")
            self._log(">> [Segmentasi] Otsu Thresholding diterapkan (objek buah dipisah dari latar).")

    # -----------------------------------------------------------
    #  LAPISAN 2 - EKSTRAKSI FITUR (bentuk & warna)
    # -----------------------------------------------------------
    def _ekstrak_fitur_objek(self, cnt, citra_bgr):
        """F1 Hue dominan, F2 Luas area, F3 Aspect ratio, F4 Solidity."""
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        luas_hull = cv2.contourArea(hull)
        solidity = float(area) / luas_hull if luas_hull > 0 else 0

        (_, _), (lebar, tinggi), _ = cv2.minAreaRect(cnt)
        sisi_panjang = max(lebar, tinggi)
        sisi_pendek = min(lebar, tinggi) if min(lebar, tinggi) > 0 else 1
        aspect_ratio = sisi_panjang / sisi_pendek

        mask = np.zeros(citra_bgr.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        hsv = cv2.cvtColor(citra_bgr, cv2.COLOR_BGR2HSV)

        # Hue dominan diambil dari MODUS histogram (bukan rata-rata aritmatika).
        # Rata-rata biasa mudah "tertarik" oleh piksel minoritas (mis. daun/
        # tangkai hijau kecil yang ikut masuk kontur, atau noise di tepi
        # objek akibat kompresi JPEG) dan juga rawan salah saat hue berada
        # di sekitar titik lingkar 0/179 (warna merah). Modus histogram jauh
        # lebih tahan terhadap gangguan semacam itu.
        #
        # Perbaikan tambahan untuk foto ASLI (bukan sintetis): area kontur
        # sering ikut memuat piksel nyaris putih/abu-abu (pantulan cahaya
        # kuat, bagian daging buah yang terpotong, bayangan tipis di tepi).
        # Piksel seperti ini punya Saturation/Value rendah sehingga nilai
        # Hue-nya sebenarnya tidak bermakna (noise), tetapi tetap ikut
        # "memilih" ke arah Hue=0 di OpenCV. Maka sebelum menghitung modus,
        # piksel dengan Saturation atau Value terlalu rendah dibuang dulu
        # dari mask, supaya modus Hue benar-benar mewakili warna kulit buah.
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask_warna = mask.copy()
        mask_warna[(sat < 60) | (val < 40)] = 0
        if cv2.countNonZero(mask_warna) < 50:
            mask_warna = mask  # fallback jika objek memang nyaris tak berwarna

        hist_hue = cv2.calcHist([hsv], [0], mask_warna, [180], [0, 180])
        hue_dominan = float(np.argmax(hist_hue))

        return {"hue": hue_dominan, "area": area, "aspect_ratio": aspect_ratio, "solidity": solidity}

    def ekstraksi_fitur(self):
        if self.cek_gambar():
            mask = self._segmentasi_mask(self.gambar_asli)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hasil = self.gambar_asli.copy()
            self._log(">> [Ekstraksi Fitur] Menghitung F1(Hue), F2(Luas), F3(Aspect Ratio), F4(Solidity):", bersihkan=True)

            jumlah = 0
            for idx, cnt in enumerate(contours, start=1):
                if cv2.contourArea(cnt) < 500:
                    continue
                jumlah += 1
                f = self._ekstrak_fitur_objek(cnt, self.gambar_asli)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(hasil, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(hasil, f"Obj{idx}", (x, max(y - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

                self._log(f"   Objek {idx} -> Hue:{f['hue']:.1f}  Luas:{int(f['area'])}px  "
                          f"AspectRatio:{f['aspect_ratio']:.2f}  Solidity:{f['solidity']:.2f}")

            self.gambar_hasil = hasil
            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)
            self.status_var.set(f"Ekstraksi Fitur selesai - {jumlah} objek terukur (lihat panel Log Hasil).")

    # -----------------------------------------------------------
    #  LAPISAN 3 - PENGENALAN POLA (MENU UTAMA)
    #  Pipeline: Segmentasi -> Ekstraksi Fitur -> Klasifikasi Rule-Based
    # -----------------------------------------------------------
    def pengenalan_pola(self):
        if self.cek_gambar():
            mask = self._segmentasi_mask(self.gambar_asli)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hasil = self.gambar_asli.copy()

            daftar_hasil = []
            self._log(">> [Pengenalan Pola] Hasil klasifikasi buah:", bersihkan=True)

            for idx, cnt in enumerate(contours, start=1):
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue
                f = self._ekstrak_fitur_objek(cnt, self.gambar_asli)
                nama_kelas, warna_kotak = klasifikasi_buah(
                    f["hue"], f["area"], f["aspect_ratio"], f["solidity"]
                )

                x, y, w, h = cv2.boundingRect(cnt)
                cv2.drawContours(hasil, [cnt], -1, warna_kotak, 3)
                cv2.rectangle(hasil, (x, y), (x + w, y + h), warna_kotak, 2)
                cv2.putText(hasil, nama_kelas, (x, max(y - 10, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, warna_kotak, 2, cv2.LINE_AA)

                daftar_hasil.append({"objek": idx, "kelas": nama_kelas, **f})
                self._log(f"   Objek {idx} -> {nama_kelas}  "
                          f"(Hue:{f['hue']:.1f} Luas:{int(f['area'])}px "
                          f"AR:{f['aspect_ratio']:.2f} Solid:{f['solidity']:.2f})")

            self.gambar_hasil = hasil
            self.daftar_hasil_klasifikasi = daftar_hasil
            self.tampilkan_gambar(self.gambar_hasil, self.label_hasil)

            jumlah_objek = len(daftar_hasil)
            if jumlah_objek == 0:
                self.status_var.set("Pengenalan Pola: tidak ada objek buah terdeteksi pada citra ini.")
                self._log("   Tidak ada objek buah terdeteksi.")
            else:
                ringkas = ", ".join(d["kelas"] for d in daftar_hasil)
                self.status_var.set(f"Pengenalan Pola selesai - {jumlah_objek} objek terdeteksi ({ringkas}).")

    # -----------------------------------------------------------
    #  SIMPAN HASIL
    # -----------------------------------------------------------
    def simpan_hasil(self):
        if self.gambar_hasil is None:
            messagebox.showwarning("Peringatan", "Belum ada citra hasil yang bisa disimpan!")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg"), ("BMP", "*.bmp")]
        )
        if path:
            cv2.imwrite(path, self.gambar_hasil)
            messagebox.showinfo("Berhasil", "Citra hasil berhasil disimpan!")
            self.status_var.set(f"Citra hasil disimpan ke: {os.path.basename(path)}")
            self._log(f">> Citra hasil disimpan ke: {os.path.basename(path)}")


# ==================================================
# RUN PROGRAM
# ==================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AplikasiPCD(root)
    root.mainloop()