# Handoff M2 Ciringoni

## Mulai di sini

**E2 telah berhasil dievaluasi dari gambar dan checkpoint di Kaggle Versi 4.** Fokus lanjutan adalah membuat alur E3 dan E4 sama jelas dan dapat diulang, lalu menjelaskan tujuan serta hasil tiap eksperimen kepada dosen. Prioritas program dan Kaggle mendahului pembaruan laporan.

```sh
git clone --branch codex/handoff-m2-kaggle-program --single-branch https://github.com/Jordyee/ciringoni-data-mining.git
cd ciringoni-data-mining
```

Buka [notebook E2](m2/kaggle/e2_validasi_kaggle.ipynb), [CSV Kaggle V4](m2/hasil/e2_kaggle_v4.csv), dan [catatan run](m2/hasil/run_e2_v4.json). Snapshot kode berasal dari notebook Kaggle V4 yang sudah selesai. Output sel dikosongkan pada salinan GitHub agar pembaca menjalankan input yang terpasang sendiri.

## Tujuan dan hasil yang sudah dapat diperiksa

Pada 356 gambar validasi dan 2.205 kotak acuan, uji apakah resolusi native 1280 meningkatkan deteksi area kecil kerusakan jagung berlabel `infected` dibandingkan native 640. Kedua checkpoint berasal dari pelatihan lima epoch, batch 4 dan seed 42 pada pasangan sumber yang sama. Run Kaggle menjalankan inferensi dan evaluasi ulang checkpoint tersebut.

| Hasil V4 CPU Kaggle | Native 640 | Native 1280 |
| --- | ---: | ---: |
| AP-small COCO | 0,0218720143 | 0,0426235588 |
| Recall confidence 0,05 dan IoU 0,50 | 0,2793650794 | 0,3628117914 |
| mAP50–95 Ultralytics | 0,0198789750 | 0,0326302065 |

AP-small naik 0,0207515444. Kandidat sementara M2 adalah native 1280 gambar penuh. Angka ini berbeda sedikit dari hasil lokal yang tersimpan. V4 memakai CPU Kaggle, Python 3.13.18 dan PyTorch 2.11.0+cpu, sedangkan lingkungan lokal acuan memakai PyTorch 2.9.1 dan skor GPU. Penyebab numerik perbedaan masih perlu diperiksa sebelum mengklaim kesamaan sempurna.

Latensi lokal dalam ringkasan historis berasal dari perangkat dan prosedur tersendiri. Run V4 belum mengukur benchmark latensi CPU Kaggle.

## Akses dan alur Kaggle

[Notebook privat M2](https://www.kaggle.com/code/davejordygerungan/fp-data-mining-m2-ciringoni) · [Dataset privat validasi E2](https://www.kaggle.com/datasets/davejordygerungan/ciringoni-m2-validasi-e2) · [Output notebook](https://www.kaggle.com/code/davejordygerungan/fp-data-mining-m2-ciringoni/output)

Pilih **Version 4** pada riwayat notebook untuk mencocokkan hasil di branch ini. Status V4 COMPLETE diperiksa pada 18 November 2026. V1 berisi rekap angka dalam kode. V2 sempat tersimpan sebelum input siap. V3 memakai GPU dan saat pemeriksaan handoff juga berstatus COMPLETE, tetapi angka V3 belum diaudit dalam pembaruan ini. Gunakan V4 untuk hasil yang didokumentasikan.

Pemilik akun memberi akses notebook dan dataset kepada akun Kaggle teman lewat pengaturan berbagi privat. Alternatifnya gunakan sesi akun pemilik dengan otorisasi pemilik. Clone GitHub memberikan kode dan metrik agregat. Dataset privat menyediakan gambar, label, dua bobot E2, geometri, GT serta skrip evaluator yang diperlukan notebook.

Alur V4 berjalan sebagai berikut.

1. Kaggle memasang Ultralytics 8.4.210 dan pycocotools.
2. Notebook membaca input privat, menyiapkan direktori kerja Kaggle, dan memeriksa 356 gambar serta 2.205 anotasi.
3. Kedua checkpoint memprediksi gambar validasi. Evaluator menyelaraskan kotak native ke koordinat acuan 640.
4. COCOeval menghitung AP-small. Presisi dan recall ambang tetap dihitung pada confidence 0,05 dan IoU 0,50. mAP50–95 memakai validator Ultralytics.
5. Kaggle menyimpan `E2_control_native_640/coco_val_predictions.json`, `E2_native_1280/coco_val_predictions.json`, JSON metrik masing-masing, dan `hasil_e2_kaggle.csv` di Output.

Untuk mengulang, gunakan kode notebook dan metadata dalam `m2/kaggle`, pasang input privat yang tercantum, lalu Save & Run All. Catat versi baru dan pastikan seluruh sel selesai. Instalasi library besar dan eksekusi utama dikerjakan di Kaggle. Pengguna mengizinkan lokal untuk persiapan berkas, kendali Kaggle, verifikasi hasil, dan GPU lokal bila diperlukan.

## Pekerjaan berikutnya per eksperimen

| Eksperimen | Status | Langkah berikut dan bukti selesai |
| --- | --- | --- |
| E2 resolusi | Inferensi dan evaluasi ulang lengkap di V4. | Tampilkan contoh gambar dengan kotak acuan dan prediksi dua resolusi. Periksa selisih terhadap acuan. Simpan versi dependensi dan hash checkpoint. |
| E3 pelatihan ubin | Hasil lokal tersedia, AP-small kontrol 0,01429 vs ubin 0,00339. | Tambahkan data olahan validation 640 dan dua last.pt E3 terverifikasi ke input privat. Jalankan evaluasi kedua model pada 356 gambar utuh di Kaggle. Periksa 1.240 ID latih dan 620 langkah optimizer dari log sebelum menjelaskan kontrol eksperimen. |
| E4 inferensi SAHI | Hasil lokal tersedia, AP-small penuh 0,04265 vs SAHI 0,03899. | Gunakan checkpoint E2 native 1280 yang sama pada input privat. Tambahkan evaluasi_e4_sahi.py dan SAHI 0.14.0. Smoke dua gambar, kemudian 356 gambar. Simpan prediksi, metrik dan contoh salah deteksi. |
| Latensi validasi | Acuan lokal E2/E4 tersedia. | Benchmark pasangan pada CPU dan jumlah thread yang sama setelah pemanasan. Catat batas waktu yang diukur, median dan p95. E3 masih memerlukan pengukuran. |

Skrip acuan ada di [hasil_m2](Data_Mining_Ciringoni_M1/hasil_m2). Evaluator bergantung pada fungsi AP-small dalam sel `m1-11` dari notebook M1. Dataset privat E2 membawa snapshot skrip yang dipakai run V4. Periksa perbedaan sebelum mengganti snapshot dengan kode repo. Untuk E3, minta dua checkpoint beserta log terverifikasi dari pemilik komputer asal. Input E2 saat ini hanya membawa validation, sehingga training baru memerlukan tambahan input train.

## Pemeriksaan dosen dan langkah setelahnya

Tunjukkan pertanyaan eksperimen, gambar input dan label, identitas checkpoint, prediksi pada gambar itu, kemudian skor dari 356 gambar. Buka tab Input dan Output Kaggle untuk menelusuri berkas. Jelaskan kontrol tiap perlakuan dan alasan kandidat dipilih.

Lengkapi E3 dan E4 sebelum menyatakan M2 sudah berjalan penuh di Kaggle. Jika perlu eksperimen tambahan, nyatakan hipotesis dan pembandingnya, misalnya seed tambahan E2 pada anggaran sama untuk memeriksa kestabilan. Pemilihan model memakai validation. Setelah hasil diverifikasi, perbarui laporan, nomor versi Kaggle dan bukti administrasi aktual. [Ringkasan lokal historis](RINGKASAN_M2.md)

## Prompt siap salin untuk Codex teman

> Lanjutkan M2 Ciringoni dari branch codex/handoff-m2-kaggle-program. Baca HANDOFF_M2.md dan hasil E2 Kaggle V4. Prioritaskan program serta alur input → prediksi → evaluasi → output di notebook privat Kaggle M2. E2 V4 sudah COMPLETE dengan 356 gambar dan 2.205 anotasi. Jangan menganggap perlu training ulang untuk mengevaluasi checkpoint yang tersedia. Periksa akses input privat, hash checkpoint dan environment. Audit contoh visual serta selisih hasil E2 terhadap acuan. Lanjutkan E3 dengan dua checkpoint last.pt terverifikasi dan validasi gambar utuh yang sama. Lanjutkan E4 penuh vs SAHI dengan best.pt E2 1280 yang sama. Jalankan instalasi library besar di Kaggle. Siapkan dataset privat tambahan bila E3 memerlukannya, dengan akses pemilik. Pantau Save & Run All sampai selesai, periksa JSON prediksi dan CSV hasil. Dokumentasikan apa yang benar-benar dieksekusi, versi, batasan dan tugas yang tersisa. Bantu saya memahami tujuan, kontrol dan hasil tiap eksperimen sebelum memperbarui laporan. Gunakan validation untuk perbandingan M2 dan pertahankan berkas privat sesuai izin pemilik.
