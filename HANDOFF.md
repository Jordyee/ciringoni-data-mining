# Handoff Ciringoni

## Status

- M1 memakai 1.778 gambar dan 13.904 kotak dengan split 1.244 train, 356 validation, dan 178 test. [Metrik M1](Data_Mining_Ciringoni_M1/hasil_m1/eksperimen.csv)
- E2 memulihkan 1.600 citra asli untuk train dan validation. Ekspor 640 dan 1280 memakai ID, label, dan split yang sama. Resolusi 1280 menghasilkan AP-small 0,0427 dan recall 0,3633 pada confidence 0,05 dan IoU 0,50. Kontrol 640 menghasilkan 0,0219 dan 0,2798. [Metrik E2 dan benchmark CPU](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_e2_val.csv)
- E3 training ubin pada data olahan 640, E5 head P2, dan E6 tanpa mosaic telah diuji. [Metrik eksperimen](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_mosaic_val.csv)
- E4 SAHI native memakai bobot E2 1280 yang sama dan menambah latensi serta menurunkan AP-small. E7 FP32/INT8 sudah diuji pada kontrol 640. Ekspor kandidat final dan evaluasi test masih tertunda.

## Data dan reproduksi

Jalankan dari direktori proyek. Tempatkan `UTS_DM_kelompok_ciringoni_hasil.zip` dan `yolo26n.pt` di `data/` sesuai dokumentasi tim. Ekstrak folder `dataset` dari ZIP untuk dataset YOLO olahan 640. Sesuaikan `path` di YAML dengan lokasi lokal. Untuk bobot awal gunakan [YOLO26n resmi](https://docs.ultralytics.com/models/yolo26/).

1. **Ground truth.** Di [notebook lokal M1](Data_Mining_Ciringoni_M1/FP_Data_Mining_M1_Eksekusi_Lokal.ipynb), jalankan bagian persiapan input hingga bagian 4. Pastikan `hasil_m1/coco_val_gt.json` memuat 356 gambar dan 2.205 anotasi. Skrip evaluasi juga membaca fungsi penilaian dari notebook ini.
2. **E2.** Jalankan `pulihkan_sumber_e2.py <zip-uts> <cache-raw> --workers 8 --seconds 1500`, lalu `bangun_pasangan_e2.py <zip-uts> <cache-raw> <native-root>`. Pastikan audit menghasilkan `ready=true`. Gunakan `train_e2_native.py <native-root> --batch 4` setelah memeriksa memori dengan `--smoke-only`. Gunakan `evaluasi_e2.py <checkpoint> <data.yaml> <audit.json> <output-baru> --size <640-atau-1280>`.
3. **E3–E6.** Gunakan `uji_tiling_640.py` dan `train_e3_640.py` untuk pilot E3, `evaluasi_e4_sahi.py` untuk E4 native, serta `train_mosaic.py` dan `evaluasi_mosaic.py` untuk eksperimen 640. Catat checkpoint, ukuran input, versi dependensi, dan konfigurasi tiap run.
4. **E7.** `uji_ekspor_e7_640.py` mereproduksi pilot OpenVINO pada kontrol 640. Gunakan data train untuk kalibrasi INT8 dan validation untuk membandingkan FP32 dengan INT8.

Skrip ada di [hasil_m2](Data_Mining_Ciringoni_M1/hasil_m2). Lingkungan teruji memakai Windows, Python 3.13, Ultralytics 8.4.210, PyTorch 2.9.1, torchvision 0.24.1, dan pycocotools. Pemulihan data memakai Python 3.11, Pillow, dan Windows HEIF Decoder. SAHI memakai versi 0.14.0. Untuk benchmark, jalankan proses CPU secara eksklusif.

## Lanjutkan dari kandidat E2

1. Cocokkan artefak dan checkpoint dengan hasil di repo sebelum mengulang training. Gunakan E2 1280 gambar penuh sebagai kandidat awal.
2. Putuskan anggaran training final dan konfirmasi kandidat pada 356 gambar validation. Bandingkan ekspor FP32 dan INT8 untuk kandidat terpilih.
3. Kunci hash checkpoint, resolusi, ambang, format ekspor, dan daftar ID. Setelah itu baru evaluasi 178 gambar test.
4. Perbarui handoff dan ringkasan metrik di pull request. Pertahankan UID gambar dan split saat membangun versi dataset baru.

Pembagian kerja yang disarankan adalah data dan label, training dan ekspor, evaluasi dan latensi, tiling dan analisis kesalahan, serta koordinasi dan laporan. Dokumen administrasi dan lokasi penyimpanan internal tersedia melalui koordinasi tim.
