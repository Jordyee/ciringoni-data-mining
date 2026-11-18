# Protokol penutupan Ciringoni

Keputusan eksperimen ditetapkan pada 18 November 2026 sebelum melihat hasil training 30 epoch atau skor test. Tujuan proyek adalah mendeteksi kotak kerusakan daun jagung berlabel `infected`, dengan AP-small sebagai metrik utama.

## Audit dan anggaran

- `main` dimulai dari `e376ec25680cb6204236a527fdacc45d9a2c8e5f`. Kedua `best.pt` E2 lima epoch cocok dengan SHA-256 di [handoff](HANDOFF.md).
- ZIP UTS lolos CRC. SHA-256 arsip `2a354489909a59f67dfb2f09bd45f72ebf333c740b5715cab0cdd5c6549dbfa0`. SHA-256 bobot awal `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
- Jalankan 30 epoch per resolusi pada RTX 4060 Laptop 8 GiB. Run acuan mencatat waktu training 0,165 jam pada 1280 dan 0,097 jam pada 640 untuk lima epoch, dengan peak reserved VRAM 3.568 MiB pada 1280. Proyeksi pasangan 30 epoch sekitar 94 menit di GPU yang sama. Run dimulai dari bobot pretrained yang sama, memakai batch 4, seed 42, AdamW, `lr0=0.002`, momentum 0,9, mosaic 1,0, dan `close_mosaic=0`. Pasangan final memakai satu seed. Replikasi dengan seed kedua berada di luar anggaran run ini.
- Train dan validation tetap memakai pasangan native E2 dengan 1.244 dan 356 ID. Test dibangun di direktori terpisah dari ZIP UTS dan sumber hasil verifikasi MD5. Kedua resolusi test memiliki 178 ID dan COCO ground truth yang sama dengan 1.159 kotak di sistem koordinat UTS 640. Audit mencatat geometri serta hash setiap gambar dan label. Metadata UTS memuat 1.778 stem, ID citra, dan MD5 unik, 92 `source_id`, 91 grup, dan 325 pasangan hampir kembar yang dipertahankan. Setiap grup, sumber, dan pasangan yang tercatat berada dalam satu split.
- Audit ulang train dan validation memverifikasi MD5 1.600 sumber, kecocokan ID gambar dan label di kedua resolusi, serta 10.540 kotak train dan 2.205 kotak validation. MAE pencocokan geometri pada referensi 640 minimum 0,2494, median 1,5655, dan maksimum 4,6978. COCOeval ulang dari prediksi E2 acuan menghasilkan AP-small 0,021894 / 0,042689 dan COCO mAP50–95 0,023358 / 0,036114 untuk 640 / 1280. Angka mAP dari COCOeval dan Ultralytics berasal dari prosedur yang berbeda.
- Audit test berhasil untuk semua 178 JPEG pada masing-masing resolusi beserta labelnya. MAE geometri minimum 0,7611, median 1,6943, dan maksimum 4,9133 pada referensi 640. Sebanyak 153 sumber test memiliki sisi terpanjang minimal 1.280 piksel. Ground truth COCO mengikuti pemotongan ke batas citra dari notebook M1 untuk 13 kotak yang melampaui batas hingga 0,0016 piksel akibat pembulatan UTS. Semua JPEG dan label native tetap memiliki hash yang sama setelah perubahan ground truth. Dua pembangunan ulang menghasilkan SHA-256 audit `25fccb5002d7484559ee229930984e53c7bf005bcd0d75666a5ebf88d1d37572` dan SHA-256 ground truth `3ddd5121eaf64f66ee4eae6e61e7d80cc23ee4723fcce4c1cbe143a0ff0fb564`. Semua 1.159 kotak COCO memiliki luas positif dan berada di dalam bingkai. Pemeriksaan ini hanya membangun data dan menghitung hash.
- Pemeriksaan tambahan pada seluruh 1.778 ekspor native 640 menemukan nol JPEG identik lintas split dan nol pasangan lintas split dengan jarak dHash 64 bit ≤4 pada area isi citra. Audit visual memeriksa sampel terarah 40 train dan 20 validation dengan 907 kotak, termasuk 71 dengan sisi terpendek <8 px. Contoh batas daun bertumpuk dan bercak samar menunjukkan bahwa kelengkapan label tetap perlu penilaian manusia. Sampel terarah dan dHash tidak dapat menetapkan identitas tanaman atau sesi pemotretan.

## Seleksi pada validation

1. Bandingkan `best.pt` native 1280 gambar penuh dengan kontrol native 640 gambar penuh pada anggaran 30 epoch. Lanjutkan 1280 jika selisih AP-small minimal +0,005, selisih COCO mAP50–95 minimal −0,005, dan selisih recall minimal −0,02. Gunakan kontrol 640 jika 1280 gagal memenuhi syarat. Lima epoch merupakan acuan untuk perubahan anggaran. Data, geometri dan 356 ID validation tetap sama.
2. Ekspor kedua varian OpenVINO dari satu checkpoint terpilih dengan input statis, batch 1, dan head end-to-end. Kalibrasi INT8 memakai 320 gambar dari train, diambil dari daftar train terurut dengan seed 42. Skor kedua varian pada 356 ID validation dengan protokol yang sama dan catat ukuran XML + BIN.
3. Pilih INT8 jika penurunan AP-small dan COCO mAP50–95 masing-masing paling banyak 0,005, penurunan recall paling banyak 0,02, dan p95 CPU lebih rendah daripada FP32. Jika syarat ini gagal, kunci FP32. Kedua ekspor diuji pada CPU yang sama dengan afinitas empat CPU logis, empat thread PyTorch, dan pemanasan 10 gambar.
4. Simpan lock yang mencakup SHA-256 checkpoint, hasil validation ekspor terpilih, XML, BIN, metadata, arsip UTS, audit test, ground truth, daftar ID, evaluator, versi pustaka, resolusi, format, presisi, dan ambang. Jalankan test hanya menggunakan konfigurasi terkunci. Gunakan satu hasil test ini untuk laporan.

AP memakai IoU 0,50–0,95, confidence minimum 0,001 dan maksimal 100 prediksi per gambar. AP-small mengikuti luas kotak pada sistem koordinat UTS 640. Presisi dan recall dihitung pada confidence 0,05 dan IoU 0,50. Inferensi memakai head end-to-end dengan satu gambar persegi per input. Latensi mencakup prapemrosesan, inferensi, pascapemrosesan, dan proyeksi koordinat balik. Waktu baca JPEG berada di luar pengukuran. [Protokol COCO](https://cocodataset.org/dataset/detection-eval.htm)

## Reproduksi

Jalankan dari `Data_Mining_Ciringoni_M1`. Gunakan Python 3.13 dengan PyTorch 2.9.1+cu126 dan torchvision 0.24.1+cu126 untuk training CUDA. Gunakan build CPU PyTorch 2.9.1 dan torchvision 0.24.1 untuk evaluasi final. Lingkungan evaluasi memakai Ultralytics 8.4.210, pycocotools 2.0.11, OpenVINO 2026.4.1, NNCF 3.4.0, NumPy 2.4.6, dan OpenCV 5.0.0.93. Pemulihan dan ekspor data memakai Python 3.11, Pillow, dan Windows HEIF Decoder. Ganti placeholder dengan path lokal dan pilih direktori keluaran baru. [Petunjuk OpenVINO Ultralytics](https://docs.ultralytics.com/integrations/openvino)

```powershell
python hasil_m2/train_e2_native.py <native-train-val> --batch 4 --epochs 30 --output <run-final>
py -3.11 hasil_m2/pulihkan_sumber_e2.py <zip-uts> <cache-test> --test-only --workers 4 --seconds 650
py -3.11 hasil_m2/bangun_test_native.py <zip-uts> <cache-test> <native-test>
python hasil_m2/evaluasi_final.py val <checkpoint-640> <native-train-val> <validasi-640> --size 640
python hasil_m2/evaluasi_final.py val <checkpoint-1280> <native-train-val> <validasi-1280> --size 1280
python hasil_m2/ekspor_final.py <checkpoint-terpilih> <native-train-val> <ekspor> --size <640-atau-1280>
python hasil_m2/evaluasi_final.py lock <folder-model-OpenVINO> <native-test> <lock.json> --size <resolusi> --precision <fp32-atau-int8> --checkpoint <checkpoint-terpilih> --validation-metrics <metrics.json-varian>
python hasil_m2/evaluasi_final.py test <folder-model-OpenVINO> <native-test> <hasil-test> --size <resolusi> --precision <fp32-atau-int8> --checkpoint <checkpoint-terpilih> --validation-metrics <metrics.json-varian> --lock <lock.json>
```

Verifikasi jalur inferensi sebelum run final memakai 12 gambar validation, model 1280 lima epoch, dan kalibrasi smoke INT8 pada 16 gambar train. Kedua IR statis berinput 1 × 3 × 1280 × 1280 dan menjalankan head end-to-end. IR INT8 memuat 125 operasi `FakeQuantize`, sedangkan IR FP32 memuat nol. Kedua backend OpenVINO melaporkan enam thread worker dengan afinitas proses pada empat CPU logis. Smoke memeriksa jalur eksekusi dan bentuk graph.

Artefak tim menyimpan manifest, prediksi per gambar, kurva, log, bobot, paket ekspor, lock lengkap, dan bahan pengumpulan. Repo publik menyimpan protokol, skrip, dan metrik agregat.

## Training dan seleksi checkpoint

Kedua run menyelesaikan 30 epoch dengan batch aktual 4. Log mencatat 0,899 jam dan peak reserved VRAM 3.568 MiB untuk 1280, serta 0,555 jam dan 1.052 MiB untuk 640. Siklus penuh berlangsung dari 06.19 hingga 07.47 WIB pada 18 November 2026. SHA-256 `best.pt` 1280 adalah `b8cc53299bf4476bf9492e286fd3c6d8422813b9701b972b641c5e1d124ff51c`. SHA-256 `best.pt` 640 adalah `abf824a20c1bd42b216a09882afe94561d271aec72a404a73a39b21ebd4629ee`. Kedua folder run memuat `args.yaml`, `results.csv` dengan 30 baris, kurva, dan bobot.

| PyTorch pada 356 gambar validation | AP-small | COCO mAP50–95 | Presisi | Recall | CPU median/p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 640, 5 epoch | 0,01583 | 0,01680 | 0,13533 | 0,12698 | 53,16 / 61,67 |
| 1280, 5 epoch | 0,03081 | 0,02597 | 0,16022 | 0,20091 | 200,54 / 224,90 |
| 640, 30 epoch | 0,04441 | 0,03718 | 0,18864 | 0,22902 | 55,63 / 63,58 |
| 1280, 30 epoch | 0,05913 | 0,05135 | 0,14684 | 0,39909 | 185,78 / 219,64 |

Kandidat 1280 memenuhi ambang untuk ketiga metrik keputusan. Selisih terhadap kontrol 640 pada 30 epoch adalah +0,01472 AP-small, +0,01418 mAP50–95, dan +0,17007 recall. Presisi pada ambang tetap turun 0,04181, dan p95 CPU naik dari 63,58 ke 219,64 ms. Pembandingan empat baris di atas memakai `nms=False` untuk mengunci head end-to-end yang juga digunakan saat ekspor. Hasil E2 historis di awal dokumen memakai pengaturan `nms` bawaan Ultralytics saat inferensi. Angka historis dan hasil inferensi ulang di tabel berasal dari protokol yang berbeda.

Training berjalan di Windows build 26100 dengan driver NVIDIA 560.94 dan Python 3.13.1. Benchmark menggunakan Intel Core i7-13650HX dengan afinitas proses pada empat CPU logis. Pembuatan data memakai Python 3.11.9, Pillow 12.0.0, libjpeg 8.0, dan Windows HEIF Image Extension 1.2.62.0. Clone GitHub bersih pada `ba85421` membangun ulang audit dan ground truth test dengan hash identik dari arsip UTS dan cache sumber yang sama.

## Seleksi format dan penguncian test

E7 memakai `best.pt` native 1280 dari run 30 epoch. Kalibrasi NNCF memproses 320 dari 320 gambar train. SHA-256 daftar stem kalibrasi terurut adalah `9d2719e7864244c8e56fae04f0f88d1bc39e6b2a196bf6f63905d07c851f68ca`. Kedua IR statis berukuran 1 × 3 × 1280 × 1280. Pemeriksaan graph menemukan 0 operasi `FakeQuantize` pada FP32 dan 125 pada INT8. OpenVINO melaporkan enam thread pada kedua varian dengan afinitas proses CPU logis 0–3 dan float32 sebagai precision hint.

| OpenVINO pada 356 gambar validation | AP-small | COCO mAP50–95 | Presisi | Recall | CPU median/p95 (ms) | XML + BIN (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 0,05913 | 0,05135 | 0,14684 | 0,39909 | 162,73 / 174,93 | 10,342 |
| INT8 | 0,06000 | 0,05153 | 0,13793 | 0,39909 | 83,41 / 91,79 | 3,660 |

INT8 memenuhi batas yang ditetapkan. Selisih terhadap FP32 adalah +0,00087 AP-small, +0,00018 mAP50–95, dan 0 recall. Presisi turun 0,00891 pada ambang tetap. P95 turun 83,14 ms dan jumlah byte XML + BIN turun 64,6%. Selisih skor yang kecil diukur pada satu split dan satu checkpoint. [Konfigurasi dan fingerprint yang dikunci sebelum test](test-lock.json)

Lock dibuat pada 18 November 2026 pukul 08.14 WIB dengan SHA-256 `943b783cd09edf11dcf68120fccf3377f2d330a8c623809816e1f84dfa5d8d32`. Pilihan yang dikunci adalah YOLO26n native 1280 gambar penuh, OpenVINO INT8, batch 1, head end-to-end, confidence AP 0,001, confidence presisi/recall 0,05, IoU pencocokan 0,50, dan maksimal 100 deteksi per gambar.

## Hasil test final

Konfigurasi dikunci dan commit [d074945](https://github.com/Jordyee/ciringoni-data-mining/commit/d074945125180013d48bc6e2146aa4d662471698) terverifikasi di branch GitHub sebelum inferensi test. Evaluator mencocokkan hash checkpoint, tiga berkas OpenVINO, daftar ID, ground truth, audit dan versi kode, lalu memproses 178 gambar dan mencatat 178 latensi. [CSV final dan angka presisi penuh](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_final.csv)

| Split test | AP-small | COCO mAP50–95 | AP50 | Presisi | Recall | CPU median/p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 178 gambar, 1.159 kotak | 0,03019 | 0,03606 | 0,10732 | 0,10829 | 0,37964 | 104,21 / 119,41 |

Pada confidence 0,05 dan IoU 0,50, 440 prediksi cocok dengan anotasi, 3.623 prediksi tidak memiliki pasangan, dan 719 kotak terlewat. Model menemukan 16 dari 69 kotak dengan sisi terpendek <8 px dan 424 dari 1.090 kotak lainnya. Batas sisi 8 px merupakan profil kesalahan tambahan. AP-small tetap mengikuti luas kotak COCO. Contoh visual dari prediksi tersimpan menunjukkan satu deteksi cocok, satu kotak kecil terlewat, dan satu prediksi tanpa pasangan.

AP-small test lebih rendah daripada AP-small INT8 pada validation, 0,03019 dibanding 0,06000. Satu seed, satu koleksi citra, potensi variasi anotasi, dan keterbatasan metadata sesi membatasi generalisasi. Banyaknya kotak terlewat dan prediksi tanpa pasangan membatasi penggunaan model untuk keputusan lapangan.

Verifikasi ulang menghitung COCO AP, presisi, dan recall dari prediksi tersimpan untuk tujuh baris CSV. Nilainya identik hingga toleransi absolut `1e-12`. Verifikasi ini memakai hasil inferensi test yang sudah tersimpan. Demo terpisah berhasil memuat OpenVINO INT8 dan menyimpan prediksi pada satu gambar validation. Replikasi training penuh pada komputer lain dan pada seed tambahan belum diukur.
