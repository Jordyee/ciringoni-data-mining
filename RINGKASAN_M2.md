# Ringkasan progres Minggu ke-2 Ciringoni

Pada 356 gambar validasi dengan 2.205 kotak acuan, YOLO26n native 1280 dengan inferensi gambar penuh menjadi kandidat untuk tahap berikutnya. Ringkasan disusun kembali pada 18 November 2026. Jadwal M2 adalah 14–16 November dan hasil teknis terkonfirmasi pada 17 November.

| Eksperimen | Perlakuan | AP-small | Recall pada ambang tetap | CPU median / p95 (ms per gambar) |
| --- | --- | ---: | ---: | ---: |
| E2 | Native 640 | 0,02189 | 0,2798 | 28,6 / 39,2 |
| E2 | Native 1280 | 0,04269 | 0,3633 | 120,4 / 146,5 |
| E3 | Latih dengan gambar utuh 640 | 0,01429 | 0,2467 | Belum tercatat |
| E3 | Latih dengan ubin 384 dari gambar olahan 640 | 0,00339 | 0,1156 | Belum tercatat |
| E4 | Inferensi gambar penuh 1280 | 0,04265 | 0,3628 | 120,0 / 145,6 |
| E4 | Inferensi SAHI, ubin 640 | 0,03899 | 0,3732 | 307,7 / 379,9 |

**E2 menguji resolusi sumber.** Kedua ekspor berasal dari 1.244 gambar latih dan 356 gambar validasi yang sama. Audit memeriksa identitas sumber, split dan geometri label. Kedua model dilatih lima epoch dengan batch 4 dan seed 42. Pada 1280, AP-small dan recall naik, dengan median waktu CPU sekitar empat kali lipat. [CSV validasi E2](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_e2_val.csv)

**E3 menguji cara melatih model.** Kedua model memakai 1.240 ID latih dari data olahan 640. Empat epoch gambar utuh dan satu epoch berisi 4.960 ubin masing-masing menghasilkan 620 langkah optimizer. Pada evaluasi gambar utuh, model yang dilatih dengan ubin menghasilkan AP-small dan recall lebih rendah daripada kontrolnya. [CSV validasi E3, baris E3_control_full_640 dan E3_tile384_640](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_mosaic_val.csv)

**E4 menguji cara inferensi.** Kedua metode memakai checkpoint E2 native 1280 yang sama. SAHI menggabungkan prediksi dari sembilan ubin per gambar dengan tumpang tindih nominal 25%. Recall meningkat, tetapi AP-small turun dan median latensi CPU menjadi 2,56 kali lipat. COCO AP50–95 turun dari 0,03610 menjadi 0,03270. [Evaluator E4](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi_e4_sahi.py)

AP-small dihitung dengan COCOeval pada koordinat acuan 640. Recall dalam tabel menggunakan confidence 0,05 dan IoU 0,50. CSV E2 dan E3 juga memuat recall dan mAP50–95 dari Ultralytics. E4 memakai prediksi GPU dan evaluator COCO sendiri untuk skor, serta proses CPU terpisah untuk latensi yang mencakup penggabungan ubin dan pemetaan koordinat. Karena prosedurnya berbeda, gunakan baris kontrol dari eksperimen yang sama saat menghitung perubahan.
