# Handoff Ciringoni

## Status

- [x] **Migrasi GitHub SELESAI.** [Repo publik Ciringoni](https://github.com/Jordyee/ciringoni-data-mining), branch `main`, initial commit [714b642](https://github.com/Jordyee/ciringoni-data-mining/commit/714b6420396abe3a596534b5829f40beddbd6207). Dataset, bobot, dokumen Word, bukti internal, dan handoff internal tetap di luar Git sesuai `.gitignore`.
- [x] M1 selesai: 1.778 gambar dan 13.904 kotak dengan split 1.244 train, 356 validation, dan 178 test. [Metrik M1](Data_Mining_Ciringoni_M1/hasil_m1/eksperimen.csv)
- [x] E2 selesai: 1.600 citra asli train+validation lolos MD5 dan audit geometri. Ekspor 640/1280 memakai ID, label, dan split yang sama; lima epoch, batch 4, seed 42. AP-small 0,0219→0,0427 dan recall 0,2798→0,3633 pada confidence 0,05 dan IoU 0,50. [Metrik E2 dan benchmark CPU](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_e2_val.csv)
- [x] E3 training ubin pada data olahan 640, E5 head P2, dan E6 tanpa mosaic selesai; tidak mengalahkan kontrol masing-masing pada AP-small. [Metrik eksperimen](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_mosaic_val.csv)
- [x] E4 SAHI native selesai: bobot E2 1280 yang sama menurunkan AP-small 0,04265→0,03899, recall tetap naik 0,3628→0,3732, dan median CPU naik 120,0→307,7 ms. Angka E4 memakai protokol evaluasinya sendiri.
- [x] E7 pilot OpenVINO FP32/INT8 pada kontrol 640 selesai.
- [ ] Konfirmasi kandidat E2 native 1280 gambar penuh dan anggaran training final pada validation.
- [ ] Bandingkan ekspor FP32/INT8 kandidat final; pilot E7 kontrol 640 belum mencakup kandidat ini.
- [ ] Kunci konfigurasi dan siapkan sumber native test secara terpisah; builder E2 saat ini hanya mengolah train+validation. Evaluasi final pada 178 gambar test belum dilakukan.

## Data dan reproduksi

Setelah clone, jalankan semua perintah Python di bawah dari `ciringoni-data-mining/Data_Mining_Ciringoni_M1` (lihat langkah `cd` di [README](README.md)). Tempatkan `UTS_DM_kelompok_ciringoni_hasil.zip` dan `yolo26n.pt` di `data/` sesuai dokumentasi tim. Ekstrak folder `dataset` dari ZIP untuk dataset YOLO olahan 640. Sesuaikan `path` di YAML dengan lokasi lokal. Untuk bobot awal gunakan [YOLO26n resmi](https://docs.ultralytics.com/models/yolo26/). Ganti placeholder `<...>` dengan lokasi lokal; gunakan folder keluaran baru untuk menjaga hasil acuan.

1. **Ground truth.** Di [notebook lokal M1](Data_Mining_Ciringoni_M1/FP_Data_Mining_M1_Eksekusi_Lokal.ipynb), jalankan bagian persiapan input hingga bagian 4. Pastikan `hasil_m1/coco_val_gt.json` memuat 356 gambar dan 2.205 anotasi. Skrip evaluasi juga membaca fungsi penilaian dari notebook ini.
2. **E2.** Jalankan `python hasil_m2/pulihkan_sumber_e2.py <zip-uts> <cache-raw> --workers 8 --seconds 1500`, lalu `python hasil_m2/bangun_pasangan_e2.py <zip-uts> <cache-raw> <native-root>`. Pastikan `audit_pasangan_e2.json` menghasilkan `ready=true`. Gunakan `python hasil_m2/train_e2_native.py <native-root> --batch 4 --smoke-only` untuk memeriksa memori, lalu hilangkan `--smoke-only` untuk training. Evaluasi dengan `python hasil_m2/evaluasi_e2.py <checkpoint> <native-root>/1280/data.yaml <native-root>/audit_pasangan_e2.json <output-baru> --size 1280`; untuk kontrol, ganti checkpoint, YAML, dan `--size` menjadi 640.
3. **E3–E6.** E3 memakai `python hasil_m2/uji_tiling_640.py <processed-root> --export-train <tile-root>`, lalu `python hasil_m2/train_e3_640.py <processed-root> <tile-root>`. E4 native memakai `python hasil_m2/evaluasi_e4_sahi.py <E2-best.pt> <native-root>/1280/images/val <native-root>/audit_pasangan_e2.json <output-baru> --device cuda:0` (CPU: `--device cpu`). E0/E6 memakai `python hasil_m2/train_mosaic.py --data <data.yaml>`; tambahkan `--run E5_640_p2` untuk E5. Evaluasi dengan `python hasil_m2/evaluasi_mosaic.py <data.yaml> --runs E0_640 E6_640_no_mosaic E5_640_p2 E3_control_full_640 E3_tile384_640`. Catat checkpoint, ukuran input, versi dependensi, dan konfigurasi tiap run.
4. **E7.** `python hasil_m2/uji_ekspor_e7_640.py <checkpoint-kontrol-640> <processed-root> <output-baru>` mereproduksi pilot OpenVINO pada kontrol 640. Gunakan data train untuk kalibrasi INT8 dan validation untuk membandingkan FP32 dengan INT8.

Skrip ada di [hasil_m2](Data_Mining_Ciringoni_M1/hasil_m2). Lingkungan teruji memakai Windows, Python 3.13, Ultralytics 8.4.210, PyTorch 2.9.1, torchvision 0.24.1, dan pycocotools. Pemulihan data memakai Python 3.11, Pillow, dan Windows HEIF Decoder. SAHI memakai versi 0.14.0. Untuk benchmark, jalankan proses CPU secara eksklusif.

## Lanjutkan dari kandidat E2

1. Cocokkan artefak dan checkpoint dengan hasil di repo sebelum mengulang training. Gunakan E2 1280 gambar penuh sebagai kandidat awal.
2. Putuskan anggaran training final dan konfirmasi kandidat pada 356 gambar validation. Bandingkan ekspor FP32 dan INT8 untuk kandidat terpilih.
3. Kunci hash checkpoint, resolusi, ambang, format ekspor, dan daftar ID. Setelah itu baru evaluasi 178 gambar test.
4. Perbarui handoff dan ringkasan metrik di pull request. Pertahankan UID gambar dan split saat membangun versi dataset baru.

Pembagian kerja yang disarankan adalah data dan label, training dan ekspor, evaluasi dan latensi, tiling dan analisis kesalahan, serta koordinasi dan laporan. Dokumen administrasi dan lokasi penyimpanan internal tersedia melalui koordinasi tim.

## Catatan kemajuan

- **17 November 2026 — penutup dokumentasi.** Mencatat migrasi GitHub selesai dan checklist eksperimen; memperjelas clone, direktori kerja, perintah reproduksi, workflow satu tugas, dan ringkasan pemeriksaan E2 di README. Verifikasi: tautan berkas lokal, argumen contoh terhadap parser CLI, angka E2 terhadap CSV, `git diff --check`, dan pemeriksaan isi publik. Tidak menjalankan training, unduhan data, ekspor model, atau evaluasi test. Hash perubahan tersedia di riwayat Git.
