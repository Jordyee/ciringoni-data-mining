# Replikasi Kaggle Ciringoni

Notebook replikasi memakai data UTS dengan pembagian tetap. Input pelatihan memuat 1.244 gambar train dan 356 gambar validation. Input test memuat 178 gambar dan dipasang pada notebook test setelah konfigurasi final dikunci. Notebook, dataset, dan keluaran terperinci di Kaggle bersifat privat. GitHub menyimpan kode dan bukti agregat.

| Minggu | Eksperimen | Pertanyaan | Program dan bukti |
| --- | --- | --- | --- |
| M1 | E0, E1 | Bandingkan YOLO26n dan YOLO26s pada resolusi olahan 640. Verifikasi profil luas kotak, ekspor 1280 dan COCOeval. | [Audit M1 COMPLETE](#m1-audit-data) · [Evaluasi historis COMPLETE](#m1-evaluasi-checkpoint-historis) · [Training M1 di antrean](https://www.kaggle.com/code/davejordygerungan/fp-data-mining-m1-kaggle-rerun) |
| M2 | E2 | Bandingkan native 640 dan 1280 dengan sumber, split, bobot awal dan anggaran pelatihan yang sama. | [Evaluasi historis COMPLETE](#m2-e2-evaluasi-resolusi) · [Program training E2](e2/E2_Kaggle.ipynb) |
| M2 | E3, E4 | Ukur dampak pelatihan tile dan inferensi SAHI pada validation. | [E3 pilot COMPLETE](#m2-e3-pilot-pelatihan-tile) · [Program E3 native](e3-native/) · [E4 pilot COMPLETE](#m2-e4-pilot-sahi) |
| M3 | E5, E6 | Bandingkan head P2 dan tanpa mosaic dengan kontrol 640. Tinjau jenis kesalahan. | [Evaluasi historis COMPLETE](#m3-evaluasi-p2-dan-mosaic) · [Training M3 di antrean](https://www.kaggle.com/code/davejordygerungan/ciringoni-m3-e5-p2-and-e6-augmentation) |
| M4 | E7 dan test | Bandingkan resolusi kandidat, OpenVINO FP32 dan INT8 pada validation. Jalankan test setelah lock dipublikasikan. | [Program E7](m4-e7/) · [Kaggle E7 V1](https://www.kaggle.com/code/davejordygerungan/ciringoni-kaggle-m4-e7-reference-checkpoints) |

## M1 E0 dan E1

Kedua model menggunakan data train dan validation olahan 640, lima epoch, batch 8 dan seed 42. E1 memakai YOLO26s sesuai starter dosen. E1 historis menggunakan YOLO26n dengan 10 lapisan dibekukan. Ekspor 1280 M1 memperbesar gambar dari olahan 640. E2 memakai ekspor dari sumber asli.

Kaggle menerima Save & Run All versi 1. Metrik validation dan hash checkpoint akan dicatat setelah output diverifikasi. Berkas studi sebelumnya tersedia di [handoff](../HANDOFF.md).

## M2 E4 pilot SAHI

**Save & Run All versi 1 selesai** pada 22 November 2026 pukul 00.42 WIB. Kaggle menjalankan inferensi baru di CPU pada 356 gambar validation dengan 2.205 kotak acuan. Kedua metode memakai checkpoint YOLO26n native 1280 lima epoch dari eksperimen E2 historis. SAHI memakai ubin 640 dengan overlap 25% dan tanpa prediksi gambar penuh. [Notebook Kaggle privat](https://www.kaggle.com/code/davejordygerungan/fp-data-mining-e4-sahi-rerun) · [Program](e4-pilot/e4_sahi.ipynb) · [CSV hasil](evidence/e4-pilot/hasil_e4_kaggle_v1.csv) · [Bukti verifikasi](evidence/e4-pilot/verifikasi_kaggle_v1.json)

| Inferensi | COCO AP-small | COCO mAP50–95 | Presisi | Recall | Median CPU | p95 CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gambar penuh | 0,04262 | 0,03607 | 0,12759 | 0,36281 | 381,71 ms | 429,15 ms |
| SAHI | 0,03894 | 0,03269 | 0,11092 | 0,37324 | 861,39 ms | 1.019,48 ms |

Presisi dan recall memakai confidence 0,05 dan IoU 0,50. SAHI meningkatkan recall 0,01043 dengan penurunan AP-small 0,00368 dan peningkatan median CPU 479,68 ms pada Intel Xeon dengan empat thread. Kaggle Output menyimpan prediksi kedua metode, contoh visual, metrik asli dan manifest hash. CSV menyajikan ulang angka dari JSON metrik. [Keluaran Kaggle](https://www.kaggle.com/code/davejordygerungan/fp-data-mining-e4-sahi-rerun/output)

## M1 audit data

**Save & Run All versi 1 selesai.** Kaggle memeriksa 1.244 gambar train dan 356 gambar validation, dengan 10.540 dan 2.205 kotak. Terdapat 756 kotak train (7,17%) dan 120 kotak validation (5,44%) dengan sisi terpendek di bawah 8 px pada acuan 640. Uji COCOeval dengan prediksi sempurna menghasilkan AP-small 1,0. Kaggle menyimpan ekspor 1280 dari 1.600 gambar olahan beserta ringkasan dan anotasi. [Notebook Kaggle privat](https://www.kaggle.com/code/davejordygerungan/ciringoni-kaggle-m1-audit) · [Program](m1-audit/M1_Audit_Kaggle.ipynb) · [Ringkasan dan verifikasi](evidence/m1-audit)

## M2 E3 pilot pelatihan tile

**Save & Run All versi 1 selesai.** Kaggle menghitung ulang prediksi, metrik, dan latensi CPU pada 356 gambar validation olahan 640 dan 2.205 kotak. Kedua checkpoint historis memproses 620 langkah optimizer. Kontrol dilatih dengan gambar penuh dan perlakuan memakai tile 384 dari sumber olahan 640. Evaluasi keduanya memakai gambar utuh. [Notebook Kaggle privat](https://www.kaggle.com/code/davejordygerungan/fp-data-mining-e3-pilot-validasi) · [Program](e3-pilot/e3_pilot_validasi.ipynb) · [CSV dan bukti verifikasi](evidence/e3-pilot)

| Pelatihan historis | AP-small | mAP50–95 Ultralytics | Presisi | Recall | Median CPU | p95 CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gambar penuh | 0,01429 | 0,00839 | 0,04927 | 0,24671 | 87,57 ms | 101,66 ms |
| Tile 384 | 0,00339 | 0,00242 | 0,02648 | 0,11565 | 83,81 ms | 89,72 ms |

Presisi dan recall memakai confidence 0,05 dan IoU 0,50. Pada pilot ini, AP-small dan recall tile 384 lebih rendah. Eksperimen E3 revisi memakai tile 640 dari data native 1280 dengan overlap 20% dan visibilitas kotak minimal 50%. [Keluaran Kaggle pilot](https://www.kaggle.com/code/davejordygerungan/fp-data-mining-e3-pilot-validasi/output)

## M1 evaluasi checkpoint historis

**Save & Run All versi 1 selesai.** Kaggle CPU menjalankan inferensi, COCOeval, dan benchmark baru untuk checkpoint E0 YOLO26n dan E1 YOLO26n freeze=10 dari pelatihan 11 November. Kedua run memakai 356 gambar validation. Unduhan memuat hasil per gambar, latensi dan hash bobot. [Notebook Kaggle privat](https://www.kaggle.com/code/davejordygerungan/ciringoni-kaggle-m1-historical-validation) · [Program](m1-validation/M1_Historical_Validation.ipynb) · [CSV dan bukti verifikasi](evidence/m1-validation)

| Perlakuan historis | AP-small | mAP50–95 Ultralytics | Presisi Ultralytics | Recall Ultralytics | Median CPU | p95 CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 YOLO26n 640 | 0,02340 | 0,02221 | 0,19078 | 0,17506 | 61,39 ms | 68,25 ms |
| E1 YOLO26n freeze=10 | 0,02432 | 0,02400 | 0,19246 | 0,13424 | 61,61 ms | 67,57 ms |

M1 mengukur latensi model pada batch 1, empat thread, dan 10 gambar pemanasan. E1 historis menaikkan AP-small 0,00093 dan recall Ultralytics turun 0,04082. Pembandingan YOLO26n dan YOLO26s memakai notebook training baru.

## M3 evaluasi P2 dan mosaic

**Save & Run All versi 1 selesai.** Kaggle CPU menghitung inferensi baru, AP-small, latensi, dan analisis kesalahan dari kontrol E0, head P2 E5, dan varian tanpa mosaic E6. Tiga checkpoint berasal dari pelatihan lima epoch dengan batch 8 dan seed 42 pada 17 November. [Notebook privat](https://www.kaggle.com/code/davejordygerungan/ciringoni-kaggle-m3-checkpoint-evaluation) · [Program evaluasi](m3-validation/) · [CSV dan bukti](evidence/m3-validation/)

| Perlakuan pada 356 gambar validation | AP-small | Recall tetap | Median CPU | p95 CPU |
| --- | ---: | ---: | ---: | ---: |
| Kontrol 640 | 0,02407 | 0,30839 | 82,56 ms | 90,70 ms |
| Head P2 | 0,01717 | 0,27211 | 109,32 ms | 117,64 ms |
| Tanpa mosaic | 0,02340 | 0,27619 | 80,96 ms | 84,57 ms |

Kontrol terpilih untuk ablation 640. Kontrol mendeteksi 4 dari 120 kotak dengan sisi terpendek di bawah 8 px pada confidence 0,05 dan IoU 0,50. E5 dan E6 masing-masing mendeteksi 2 kotak pada kategori ini. [Program training baru](m3/)

## M4 persiapan validasi dan lock

Kaggle menerima Save & Run All versi 1 untuk E7 dengan dua checkpoint native 30 epoch historis. Notebook menjalankan evaluasi validation, ekspor FP32 dan INT8, dan memilih model menurut ambang di [protokol penutupan](../PROTOKOL_FINAL.md). Kalibrasi INT8 memakai 320 gambar train. Program training native baru tersedia untuk dijalankan ketika slot GPU tersedia. [Program training](m4-training/) · [Notebook E7 privat](https://www.kaggle.com/code/davejordygerungan/ciringoni-kaggle-m4-e7-reference-checkpoints)

Input test tersimpan di dataset Kaggle privat terpisah. Setelah E7 selesai, publikasikan lock dan verifikasi hash pada commit development sebelum notebook test menjalankan inferensi.

## M2 E2 evaluasi resolusi

**Save & Run All versi 2 selesai.** E2 menjalankan kembali inferensi, AP-small dan benchmark pada dua checkpoint historis lima epoch. Pada 356 gambar validation, native 640 dan 1280 menghasilkan AP-small 0,02187 dan 0,04262 serta recall 0,27937 dan 0,36281 pada confidence 0,05 dan IoU 0,50. Median CPU empat thread naik dari 100,23 ms menjadi 339,75 ms dan p95 naik dari 111,54 menjadi 382,17 ms. [Notebook Kaggle privat](https://www.kaggle.com/code/davejordygerungan/ciringoni-kaggle-e2-checkpoint-validation) · [Program](e2-validation/) · [CSV dan verifikasi](evidence/e2-validation/)

Kaggle menolak pengajuan training E2 karena kedua slot batch GPU terpakai oleh M1 dan M3 yang mengantre. Hasil versi 2 memakai checkpoint terverifikasi dari eksperimen historis.
