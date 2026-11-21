# Replikasi Kaggle Ciringoni

Notebook replikasi memakai data UTS dengan pembagian tetap. Input pelatihan memuat 1.244 gambar train dan 356 gambar validation. Input test memuat 178 gambar dan dipasang pada notebook test setelah konfigurasi final dikunci. Notebook, dataset, dan keluaran terperinci di Kaggle bersifat privat. GitHub menyimpan kode dan bukti agregat.

| Minggu | Eksperimen | Pertanyaan | Program dan bukti |
| --- | --- | --- | --- |
| M1 | E0, E1 | Bandingkan YOLO26n dan YOLO26s pada resolusi olahan 640. Verifikasi profil luas kotak, ekspor 1280 dan COCOeval. | [Program M1](m1/M1_Kaggle.ipynb) |
| M2 | E2 | Bandingkan native 640 dan 1280 dengan sumber, split, bobot awal dan anggaran pelatihan yang sama. | Menunggu eksekusi |
| M2 | E3, E4 | Ukur dampak pelatihan tile dan inferensi SAHI pada validation. | Menunggu verifikasi |
| M3 | E5, E6 | Bandingkan head P2 dan tanpa mosaic dengan kontrol 640. Tinjau jenis kesalahan. | Menunggu eksekusi |
| M4 | E7 dan test | Bandingkan resolusi kandidat, OpenVINO FP32 dan INT8 pada validation. Jalankan test setelah lock dipublikasikan. | Menunggu eksekusi |

## M1 E0 dan E1

Kedua model menggunakan data train dan validation olahan 640, lima epoch, batch 8 dan seed 42. E1 memakai YOLO26s sesuai starter dosen. E1 historis menggunakan YOLO26n dengan 10 lapisan dibekukan. Ekspor 1280 M1 memperbesar gambar dari olahan 640. E2 memakai ekspor dari sumber asli.

Versi selesai akan mencatat nomor run Kaggle, metrik validation, hash checkpoint dan bukti output. Berkas studi sebelumnya tersedia di [handoff](../HANDOFF.md).
