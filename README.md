# Ciringoni Data Mining

Final project untuk mendeteksi area kecil kerusakan daun jagung akibat Fall Armyworm (FAW) dengan YOLO26n. Dataset memakai satu kelas anotasi, `infected`.

## Hasil sementara

- **E2 resolusi** · Pada 356 gambar validation yang sama, pasangan yang dibangun dari citra asli menghasilkan AP-small **0,0219** pada 640 dan **0,0427** pada 1280. Kedua model dilatih lima epoch dengan batch 4. [Hasil E2](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_e2_val.csv)
- **E4 SAHI** · Pada checkpoint E2 1280 yang sama, AP-small turun dari **0,04265** pada gambar penuh menjadi **0,03899** pada inferensi ubin. Benchmark CPU terpisah mengukur median **120,0 ms** untuk gambar penuh dan **307,7 ms** untuk SAHI.
- **Kandidat sementara** · YOLO26n E2 1280 dengan inferensi gambar penuh. Split test berisi 178 gambar yang disimpan untuk evaluasi final.

## Mulai bekerja

1. Clone repo dan baca [HANDOFF.md](HANDOFF.md) untuk langkah data, training, dan evaluasi.
2. Jalankan notebook dan skrip dari folder `Data_Mining_Ciringoni_M1`. Ambil arsip dataset dan checkpoint dari penyimpanan tim yang aksesnya sudah diberikan. Dataset acuan dicatat di [notebook Kaggle UTS](https://www.kaggle.com/code/davejordygerungan/data-mining-mid).
3. Buat satu branch per tugas. Buka pull request berisi tujuan, perubahan, perintah yang dijalankan, dan hasil verifikasi.
