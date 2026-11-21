# Replikasi Kaggle Ciringoni

Notebook replikasi memakai data UTS dengan pembagian tetap. Input pelatihan memuat 1.244 gambar train dan 356 gambar validation. Input test memuat 178 gambar dan dipasang pada notebook test setelah konfigurasi final dikunci. Notebook, dataset, dan keluaran terperinci di Kaggle bersifat privat. GitHub menyimpan kode dan bukti agregat.

| Minggu | Eksperimen | Pertanyaan | Program dan bukti |
| --- | --- | --- | --- |
| M1 | E0, E1 | Bandingkan YOLO26n dan YOLO26s pada resolusi olahan 640. Verifikasi profil luas kotak, ekspor 1280 dan COCOeval. | [Audit M1 COMPLETE](#m1-audit-data) · [Training M1 di antrean](https://www.kaggle.com/code/davejordygerungan/fp-data-mining-m1-kaggle-rerun) |
| M2 | E2 | Bandingkan native 640 dan 1280 dengan sumber, split, bobot awal dan anggaran pelatihan yang sama. | [Program E2](e2/E2_Kaggle.ipynb) |
| M2 | E3, E4 | Ukur dampak pelatihan tile dan inferensi SAHI pada validation. | [E4 pilot COMPLETE](#m2-e4-pilot-sahi) |
| M3 | E5, E6 | Bandingkan head P2 dan tanpa mosaic dengan kontrol 640. Tinjau jenis kesalahan. | Menunggu eksekusi |
| M4 | E7 dan test | Bandingkan resolusi kandidat, OpenVINO FP32 dan INT8 pada validation. Jalankan test setelah lock dipublikasikan. | Menunggu eksekusi |

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
