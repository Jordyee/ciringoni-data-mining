# Handoff Ciringoni

## Status

- [x] **Migrasi GitHub SELESAI.** [Repo publik Ciringoni](https://github.com/Jordyee/ciringoni-data-mining), branch `main`, initial commit [714b642](https://github.com/Jordyee/ciringoni-data-mining/commit/714b6420396abe3a596534b5829f40beddbd6207). Dataset, bobot, dokumen Word, bukti internal, dan handoff internal tetap di luar Git sesuai `.gitignore`.
- [x] M1 selesai: 1.778 gambar dan 13.904 kotak dengan split 1.244 train, 356 validation, dan 178 test. [Metrik M1](Data_Mining_Ciringoni_M1/hasil_m1/eksperimen.csv)
- [x] E2 selesai: 1.600 citra asli train+validation lolos MD5 dan audit geometri. Ekspor 640/1280 memakai ID, label, dan split yang sama; lima epoch, batch 4, seed 42. AP-small 0,0219→0,0427 dan recall 0,2798→0,3633 pada confidence 0,05 dan IoU 0,50. [Metrik E2 dan benchmark CPU](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_e2_val.csv)
- [x] E3 training ubin pada data olahan 640, E5 head P2, dan E6 tanpa mosaic selesai; tidak mengalahkan kontrol masing-masing pada AP-small. [Metrik eksperimen](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_mosaic_val.csv)
- [x] E4 SAHI native selesai: bobot E2 1280 yang sama menurunkan AP-small 0,04265→0,03899, recall tetap naik 0,3628→0,3732, dan median CPU naik 120,0→307,7 ms. Angka E4 memakai protokol evaluasinya sendiri.
- [x] E7 pilot OpenVINO FP32/INT8 pada kontrol 640 selesai.
- [x] Audit dan pembangunan test native terpisah selesai. Kedua resolusi memuat 178 gambar dan 1.159 kotak yang cocok dengan split UTS.
- [x] Pasangan native 640 dan 1280 selesai 30 epoch dengan batch 4 dan seed 42. Pada 356 gambar validation, 1280 memenuhi ketiga ambang seleksi.
- [x] E7 final membandingkan FP32 dan INT8 pada validation, dengan 320 gambar train untuk INT8. Ukuran XML + BIN turun dari 10,34 ke 3,66 MB dan p95 CPU dari 174,93 ke 91,79 ms. Recall tetap 0,39909.
- [x] Lock native 1280 OpenVINO INT8 dipublikasikan pada commit `d074945125180013d48bc6e2146aa4d662471698`. Test 178 gambar menghasilkan AP-small 0,03019, COCO mAP50–95 0,03606, dan recall 0,37964. [Protokol dan hasil akhir](PROTOKOL_FINAL.md)
- [ ] Pengesahan dan dokumentasi rapat M4 diselesaikan bersama tim. Notebook M4 tersedia melalui repo untuk pengisian versi pengumpulan Kaggle.

## Data dan reproduksi

Setelah clone, jalankan semua perintah Python di bawah dari `ciringoni-data-mining/Data_Mining_Ciringoni_M1` (lihat langkah `cd` di [README](README.md)). Tempatkan `UTS_DM_kelompok_ciringoni_hasil.zip` dan `yolo26n.pt` di `data/` sesuai dokumentasi tim. Ekstrak folder `dataset` dari ZIP untuk dataset YOLO olahan 640. Sesuaikan `path` di YAML dengan lokasi lokal. Untuk bobot awal gunakan [YOLO26n resmi](https://docs.ultralytics.com/models/yolo26/). Ganti placeholder `<...>` dengan lokasi lokal; gunakan folder keluaran baru untuk menjaga hasil acuan.

1. **Ground truth.** Di [notebook lokal M1](Data_Mining_Ciringoni_M1/FP_Data_Mining_M1_Eksekusi_Lokal.ipynb), jalankan bagian persiapan input hingga bagian 4. Pastikan `hasil_m1/coco_val_gt.json` memuat 356 gambar dan 2.205 anotasi. Skrip evaluasi juga membaca fungsi penilaian dari notebook ini.
2. **E2.** Jalankan `python hasil_m2/pulihkan_sumber_e2.py <zip-uts> <cache-raw> --workers 8 --seconds 1500`, lalu `python hasil_m2/bangun_pasangan_e2.py <zip-uts> <cache-raw> <native-root>`. Pastikan `audit_pasangan_e2.json` menghasilkan `ready=true`. Gunakan `python hasil_m2/train_e2_native.py <native-root> --batch 4 --smoke-only` untuk memeriksa memori, lalu hilangkan `--smoke-only` untuk training. Evaluasi dengan `python hasil_m2/evaluasi_e2.py <checkpoint> <native-root>/1280/data.yaml <native-root>/audit_pasangan_e2.json <output-baru> --size 1280`; untuk kontrol, ganti checkpoint, YAML, dan `--size` menjadi 640.
3. **E3–E6.** E3 memakai `python hasil_m2/uji_tiling_640.py <processed-root> --export-train <tile-root>`, lalu `python hasil_m2/train_e3_640.py <processed-root> <tile-root>`. E4 native memakai `python hasil_m2/evaluasi_e4_sahi.py <E2-best.pt> <native-root>/1280/images/val <native-root>/audit_pasangan_e2.json <output-baru> --device cuda:0` (CPU: `--device cpu`). E0/E6 memakai `python hasil_m2/train_mosaic.py --data <data.yaml>`; tambahkan `--run E5_640_p2` untuk E5. Evaluasi dengan `python hasil_m2/evaluasi_mosaic.py <data.yaml> --runs E0_640 E6_640_no_mosaic E5_640_p2 E3_control_full_640 E3_tile384_640`. Catat checkpoint, ukuran input, versi dependensi, dan konfigurasi tiap run.
4. **E7.** `python hasil_m2/uji_ekspor_e7_640.py <checkpoint-kontrol-640> <processed-root> <output-baru>` mereproduksi pilot OpenVINO pada kontrol 640. Gunakan data train untuk kalibrasi INT8 dan validation untuk membandingkan FP32 dengan INT8.

Skrip ada di [hasil_m2](Data_Mining_Ciringoni_M1/hasil_m2). Lingkungan teruji memakai Windows, Python 3.13, Ultralytics 8.4.210, PyTorch 2.9.1, torchvision 0.24.1, dan pycocotools. Pemulihan data memakai Python 3.11, Pillow, dan Windows HEIF Decoder. SAHI memakai versi 0.14.0. Untuk benchmark, jalankan proses CPU secara eksklusif.

## Serah terima ke komputer lain

1. Clone versi terbaru repo dan gunakan README, handoff root, notebook, serta skrip dari GitHub sebagai acuan. Notebook di paket lanjutan dibuat sebelum audit dokumentasi terakhir; handoff internal di arsip juga mendahului migrasi GitHub.
2. Kirim `Ciringoni_Paket_Lanjutan.zip` dan `UTS_DM_kelompok_ciringoni_hasil.zip` secara terpisah melalui penyimpanan tim. Keduanya tersedia di komputer asal; ZIP UTS tidak termasuk di paket lanjutan. Paket lanjutan lolos pemeriksaan CRC dan membawa checkpoint E2 serta ground truth validation yang cocok dengan berkas lokal.
3. Dari paket lanjutan, salin artefak yang diabaikan Git ke folder proyek pada clone: checkpoint, hasil evaluasi terperinci, ground truth, dan bobot awal. Pertahankan berkas sumber dan dokumentasi versi GitHub. Pastikan `hasil_m1/coco_val_gt.json` memuat 356 gambar dan 2.205 anotasi.
4. Cache sumber, dataset native, dan `audit_pasangan_e2.json` tidak termasuk di paket. Pindahkan cache/dataset native secara terpisah atau bangun ulang mengikuti bagian reproduksi bila akses sumber tersedia. Sesuaikan path YAML dan manifest pada komputer penerima. Lanjut setelah audit `ready=true`, 1.600 pasangan, dan ID train/val cocok pada kedua resolusi.
5. Cocokkan SHA-256 checkpoint setelah dipindahkan. Lokasi relatif terhadap `Data_Mining_Ciringoni_M1`:

| Checkpoint | SHA-256 |
| --- | --- |
| `hasil_m2/eksperimen_e2_native/runs/E2_native_1280/weights/best.pt` | `0bd467aef478ebd26555074030ff653c52cf05ad5380fcbe66057b3c995f264e` |
| `hasil_m2/eksperimen_e2_native/runs/E2_control_native_640/weights/best.pt` | `56568daf2c930033058a3b0468f1357069b1110ce2a1e07e705405547f46d34f` |

## Gunakan hasil final

1. Baca [protokol penutupan](PROTOKOL_FINAL.md), [lock test](test-lock.json), dan [notebook hasil M4](Data_Mining_Ciringoni_M1/FP_Data_Mining_M4_Penutupan.ipynb).
2. Paket lokal `Ciringoni_Paket_Final_20261118.zip` siap dipindahkan melalui penyimpanan tim. Isinya mencakup model, konfigurasi, log, prediksi, bukti aktivitas, serta bahan laporan dan demo. Verifikasi isi terhadap `SHA256SUMS.txt`. Dataset test berada di folder privat `hasil_m2/penutupan_20261118/test-native`.
3. Jalankan demo pada gambar validation mengikuti `DEMO_M4.md` di paket. Gunakan baris TEST yang dikunci dari [CSV final](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_final.csv) untuk laporan dan presentasi.
4. Lengkapi kode resmi kelompok, notulen dan foto rapat M4, kontribusi, jam kerja, paraf, persetujuan ketua, dan versi pengumpulan sesuai template. Data, identitas, bukti foto, dan persetujuan tim disimpan privat.

Pembagian kerja yang disarankan adalah data dan label, training dan ekspor, evaluasi dan latensi, tiling dan analisis kesalahan, serta koordinasi dan laporan. Dokumen administrasi dan lokasi penyimpanan internal tersedia melalui koordinasi tim.

## Catatan kemajuan

- **18 November 2026 — persiapan penutupan.** Memverifikasi hash kedua checkpoint E2, ZIP UTS, 1.600 sumber train dan validation, dan geometri label. Menghitung kembali AP-small, COCO mAP50–95, presisi dan recall dari prediksi validation E2 tersimpan. Menjalankan training pasangan 30 epoch pada RTX 4060. Membangun test native terpisah dua kali dan mencocokkan hash audit. Evaluator menolak hash gambar yang diubah pada salinan audit. Jalur OpenVINO FP32/INT8 lolos smoke pada validation dengan kalibrasi smoke dari train. Rincian angka dan konfigurasi ada di [protokol](PROTOKOL_FINAL.md).
- **18 November 2026 — evaluasi final.** Menyelesaikan kedua run 30 epoch, evaluasi ulang pasangan lima dan 30 epoch, dan E7 pada 356 validation. Mengunci native 1280 OpenVINO INT8 di GitHub sebelum satu evaluasi test. Merekam prediksi dan latensi untuk setiap gambar, menghitung ulang metrik dari prediksi tersimpan, dan menguji demo pada satu gambar validation. Metrik agregat, lock, dan notebook M4 tersedia melalui repo. [Hasil dan keterbatasan](PROTOKOL_FINAL.md)
- **18 November 2026 — serah terima M4.** Menjalankan sel notebook M4 pada clone baru dan mencocokkan CSV, konfigurasi, serta hash lock. Memeriksa enam halaman DOCX/PDF berdasarkan template dan enam slide presentasi. Paket privat lolos CRC dan SHA-256 per entri untuk 130 berkas. SHA-256 paket `c04eaeca6c82e4552218b81a640679f8b978baf204de158e2453c021ddfb2847`. Tim melengkapi bukti rapat, persetujuan, dan versi pengumpulan.

- **17 November 2026 — penutup dokumentasi.** Mencatat migrasi GitHub selesai dan checklist eksperimen; memperjelas clone, direktori kerja, perintah reproduksi, workflow satu tugas, dan ringkasan pemeriksaan E2 di README. Verifikasi: tautan berkas lokal, argumen contoh terhadap parser CLI, angka E2 terhadap CSV, `git diff --check`, dan pemeriksaan isi publik. Tidak menjalankan training, unduhan data, ekspor model, atau evaluasi test. Hash perubahan tersedia di riwayat Git.
- **17 November 2026 — audit notebook publik.** Memeriksa dua notebook M1, termasuk metadata dan output tersimpan; tidak ditemukan NIM, tautan Drive privat, path profil komputer, atau pola kredensial yang diperiksa. Memperbaiki petunjuk lokal yang masih mengarah ke Kaggle, memperjelas konfigurasi E1 lokal yang sudah terisi, serta menautkan status terkini dan membedakan ekspor pembesaran M1 dari E2 native. Verifikasi: JSON notebook, sintaks seluruh sel kode, tautan lokal, kesamaan sel kode/metadata/output terhadap versi sebelum audit, dan `git diff --check`. Perubahan notebook hanya pada teks Markdown; notebook, training, dan test tidak dijalankan.
- **17 November 2026 — template pull request.** Menambahkan `.github/pull_request_template.md` berisi tujuan/perubahan, verifikasi aktual, dan checklist pembaruan handoff; menautkannya dari workflow README. Verifikasi: lokasi dan isi template, tautan lokal dokumen, pemeriksaan isi publik, serta `git diff --check`. Tidak menjalankan eksperimen atau mengubah model.
- **17 November 2026 — pemeriksaan serah terima.** Memeriksa sinkronisasi Git, integritas objek Git, sintaks 10 skrip dan 10 sel kode, dua notebook, 16 tautan lokal, serta pola informasi privat pada berkas teks terlacak. Paket lanjutan lolos CRC; checkpoint E2 dan ground truth di dalamnya cocok dengan berkas lokal. Audit cache native lokal `ready=true` dengan 1.600 pasangan terverifikasi. Menambahkan daftar berkas transfer, keterbatasan isi paket, dan SHA-256 checkpoint. Verifikasi statis dan integritas berkas lolos; eksekusi penuh pada komputer penerima tetap perlu diperiksa. Tidak menjalankan training atau evaluasi test.
