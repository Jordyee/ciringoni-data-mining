# Serah terima M2 Ciringoni

**Pertanyaan eksperimen.** Pada 356 gambar validasi yang sama, apakah resolusi native lebih tinggi atau penggunaan ubin memperbaiki deteksi area kecil kerusakan jagung berlabel `infected`? Ukur AP-small, recall pada confidence 0,05 dan IoU 0,50, serta median dan p95 CPU per gambar. Bandingkan perlakuan dengan kontrolnya dalam tiap eksperimen. E2 native 1280 gambar penuh adalah kandidat sementara dari pengujian lokal.

**Revisi dari pemeriksaan dosen.** Tujuan dan urutan kerja belum cukup jelas, bukti eksperimen perlu diperkuat, dan notebook Kaggle yang diperlihatkan belum menjalankan program eksperimennya. Prioritas penerus adalah program yang dapat dijalankan, input dan output yang dapat diperiksa, lalu pembaruan laporan setelah hasil terverifikasi.

## Buka dari komputer teman

```sh
git clone --branch codex/handoff-m2-kaggle-program --single-branch https://github.com/Jordyee/ciringoni-data-mining.git
cd ciringoni-data-mining
```

1. Baca [ringkasan hasil lokal](RINGKASAN_M2.md). Untuk mengulang rekap di komputer baru, buka [notebook rekap](m2/kaggle/rekap_m2.ipynb) dengan direktori kerja root repo, atau arahkan variabel lingkungan `M2_INPUT` ke root clone. Jalankan semua sel. Hasil lokal berada di `output_m2/hasil_validasi_m2.csv`. Direktori keluaran dapat diatur dengan `M2_OUTPUT`.
2. Untuk uji awal Kaggle, pemilik akun perlu memberi akses ke notebook privat atau teman dapat mengimpor notebook rekap ke akun sendiri. Pasang tiga CSV berikut sebagai satu Input: [E2](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_e2_val.csv), [E3](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_mosaic_val.csv), dan [E4](m2/kaggle/eksperimen_e4_val.csv). CSV E3 juga memuat eksperimen lain sehingga kode memilih dua ID E3. CSV E4 menyalin dua baris agregat yang sudah dibulatkan dalam ringkasan publik.
3. Jalankan **Save Version → Save & Run All**. Periksa status selesai serta unduh `/kaggle/working/hasil_validasi_m2.csv` dari Output. Harus ada enam baris dengan masing-masing 356 gambar. AP-small naik pada E2, turun pada E3 dan E4. Latensi E3 memang kosong pada sumber. Catat versi Kaggle aktual dan commit input.
4. Setelah alur CSV berjalan, lanjutkan ke program model di bawah. Bukti rekap CSV hanya membuktikan pembacaan dan perbandingan hasil tersimpan. Bukti inferensi membutuhkan checkpoint, gambar, keluaran prediksi baru dan evaluasi terhadap anotasi.

## Apa yang sudah berjalan

| Eksperimen | Bukti lokal dan program | Pekerjaan berikutnya di Kaggle |
| --- | --- | --- |
| E2 · resolusi | 1.244 train dan 356 validasi, ID dan label berpasangan. Lima epoch, batch 4, seed 42. AP-small 0,02189 → 0,04269. Alur: `bangun_pasangan_e2.py` → `train_e2_native.py` → `evaluasi_e2.py`. | Pasang dua ekspor native, audit geometri, GT dan kedua best.pt. Cek ID, jalankan inferensi dan proyeksi ke koordinat 640, ekspor prediksi lalu hitung metrik pada 356 gambar. |
| E3 · pelatihan ubin | Data olahan 640. 1.240 ID train yang sama, 620 langkah optimizer per model, kontrol utuh vs ubin 384. Validasi gambar utuh. AP-small 0,01429 → 0,00339. Alur: `uji_tiling_640.py` → `train_e3_640.py` → `evaluasi_mosaic.py`. | Pasang kedua last.pt dari run terverifikasi serta data olahan. Pilih dua ID E3 secara eksplisit saat evaluasi. Konfirmasi anggaran dari log. Latensi CPU E3 masih perlu diukur pada prosedur yang dicatat. |
| E4 · inferensi SAHI | Checkpoint E2 1280 yang sama, penuh vs sembilan ubin 640. AP-small 0,04265 → 0,03899, recall 0,3628 → 0,3732. Median CPU 120,0 → 307,7 ms. Program `evaluasi_e4_sahi.py`. | Jalankan smoke pada dua gambar, lalu kedua mode pada 356 gambar. Simpan prediksi, contoh kegagalan, skor GPU serta benchmark CPU terpisah. Catat perangkat sebelum membandingkan latensi. |

[Folder program eksperimen](Data_Mining_Ciringoni_M1/hasil_m2) · [Perintah dan hash checkpoint E2](HANDOFF.md). AP-small memakai COCOeval pada acuan 640. mAP E2/E3 dari Ultralytics dan mAP E4 dari COCOeval menggunakan prosedur berbeda. Pemilihan kandidat M2 memakai split validasi. Split test disimpan untuk tahap selanjutnya.

## Status Kaggle dan kebutuhan transfer

**Versi 1 privat**, berjudul *FP Data Mining M2 Ciringoni*, selesai dijalankan. [Salinan kode V1](m2/kaggle/arsip_v1.ipynb) memuat enam angka agregat sebagai input di dalam kode dan mencetak hasil ke log. Tidak ada dataset terpasang atau CSV keluaran di V1. Notebook rekap di branch ini merupakan persiapan versi berikutnya dan telah diuji lokal. Nomor versi berikutnya baru dicatat setelah eksekusi Kaggle selesai.

Untuk menjalankan model, minta pemilik proyek memberikan akses privat ke data olahan 640, ekspor native 640/1280 train dan val, audit pasangan yang berisi geometri per gambar, `hasil_m1/coco_val_gt.json`, dua best.pt E2 dan dua last.pt E3 beserta argumen dan log. Verifikasi ID, 2.205 kotak validasi dan hash checkpoint sebelum evaluasi. Dari komputer asal, cek isi `Ciringoni_Paket_Lanjutan.zip` terhadap daftar ini. Cache native dan audit geometri lengkap perlu dipindahkan terpisah atau dibangun ulang. Untuk training baru, diperlukan bobot awal YOLO26n.

Di Kaggle, gunakan input privat berizin di `/kaggle/input`, dan salinan kerja skrip serta keluaran di `/kaggle/working`. Sesuaikan path absolut Windows di YAML/manifest. Evaluator mengambil fungsi AP-small dari sel `m1-11` di notebook M1 lokal, jadi pertahankan struktur folder repo. Periksa dependensi serta penanganan HEIC jika membangun ulang citra native. Angka waktu CPU dapat berbeda antarperangkat.

## Urutan pemeriksaan dengan dosen

Tunjukkan pertanyaan E2, satu ID gambar dan label, checkpoint yang digunakan, prediksi yang dipetakan ke koordinat acuan, lalu satu baris hasil yang dihitung dari prediksi. Setelah jalur E2 terbukti pada 356 gambar, ulangi dengan kontrol E3 dan E4. Bilamana perlu eksperimen tambahan, sepakati hipotesis dan pembanding terlebih dahulu, misalnya menguji kestabilan E2 dengan seed tambahan pada anggaran yang sama. Simpan hasil baru terpisah dari acuan.

## Prompt untuk Codex teman

> Kerjakan M2 Ciringoni dari branch ini. Baca HANDOFF_M2.md dan RINGKASAN_M2.md. Mulai dengan menjalankan notebook rekap dari tiga CSV dan verifikasi enam baris keluarannya. Tunjukkan input, transformasi, output dan status eksekusi yang sebenarnya. Inventarisasi akses data dan checkpoint privat sebelum memakai GPU. Lalu buat satu jalur E2 dari gambar dan checkpoint sampai prediksi dan evaluasi pada 356 gambar validasi. Cocokkan hasil dengan acuan, jelaskan selisih, dan simpan versi Kaggle yang benar-benar berjalan. Lanjutkan ke E3 dengan kontrol langkah yang sama dan E4 dengan checkpoint yang sama. Usulkan eksperimen tambahan berdasarkan kekurangan yang masih terlihat. Catat per eksperimen perintah, split, versi data, hash checkpoint, versi lingkungan, waktu, output dan batasan. Jika artefak belum tersedia, sebutkan berkas spesifik yang perlu diminta. Utamakan program dan Kaggle. Perbarui laporan hanya setelah hasil diverifikasi. Jaga data, bukti, identitas dan bobot tetap privat.
