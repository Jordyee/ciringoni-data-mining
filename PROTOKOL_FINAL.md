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
