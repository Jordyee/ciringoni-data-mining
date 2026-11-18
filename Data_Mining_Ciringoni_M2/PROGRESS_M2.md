# Progres Final Project Ciringoni M2

Progres sampai 18 November 2026; verifikasi notebook Kaggle pada 19 November 2026. Dokumen ini merupakan ringkasan publik. Laporan Word/PDF, foto rapat, dan administrasi anggota disimpan dalam paket internal kelompok.

## Notebook dan sumber hasil

| Komponen | Bukti |
| --- | --- |
| Notebook Final Project | [FP Data Mining Ciringoni M2](https://www.kaggle.com/code/kaawoanprayer/fp-data-mining-ciringoni-m2), privat pada akun Prayer |
| Source Notebook di Kaggle | [Data-Mining-MID](https://www.kaggle.com/code/davejordygerungan/data-mining-mid), output UTS dipasang melalui Add Input → Notebook |
| Versi tersimpan | Versi 1, `M2 audit sumber Kaggle dan kode E2 E3 E4`, Save & Run All, Successful dalam 39,1 detik |
| Log versi tersimpan | [Kaggle versi 371146508](https://www.kaggle.com/code/kaawoanprayer/fp-data-mining-ciringoni-m2/log?scriptVersionId=371146508) dan [salinan log](bukti_analisis/log_kaggle_v1.txt) |
| Kode yang disediakan | 70 sel, termasuk 42 sel kode; [Kaggle](FP_DM_Ciringoni_Kaggle_M2.ipynb) dan [Colab](FP_DM_Ciringoni_Colab_M2.ipynb) |
| Acuan eksperimen historis | Repo pada commit `e376ec25680cb6204236a527fdacc45d9a2c8e5f`; [tabel historis](bukti_analisis/eksperimen_historis_repo.csv) |

Notebook baru meneruskan hasil prapemrosesan UTS. Versi Kaggle yang berhasil menjalankan audit dataset dan riwayat hasil dengan seluruh `LATIH=False`; versi ini tidak melatih ulang E2, E3, atau E4. Salinan Colab memiliki alur kode yang sama dan diperiksa struktur serta sintaksnya; eksekusi pada layanan Colab belum dilakukan.

## Audit dataset

| Split | Gambar | Kotak |
| --- | ---: | ---: |
| Train | 1.244 | 10.540 |
| Validation | 356 | 2.205 |
| Test | 178 | 1.159 |
| Total | 1.778 | 13.904 |

Source ID lintas split dan pasangan hampir kembar lintas split keduanya **0**. Pemeriksaan test dalam mode audit terbatas pada struktur, label, dan konsistensi split; evaluasi kinerja model pada test tetap menunggu penguncian kandidat. Sebanyak 35,62% kotak mempunyai sisi terpendek di bawah 16 piksel pada koordinat acuan 640.

Uji COCOeval dengan prediksi sempurna menghasilkan AP mendekati 1,0. Uji ini memeriksa evaluator, bukan skor model terlatih. Bukti: [audit dataset](bukti_analisis/audit_dataset.json), [profil ukuran](bukti_analisis/profil_persen_per_bin.csv), [uji evaluator](bukti_analisis/uji_prediksi_sempurna.json), dan [verifikasi notebook](bukti_analisis/pemeriksaan_notebook.json).

## Hasil E2, E3, dan E4 dari repo

| Eksperimen | Temuan | Batas interpretasi |
| --- | --- | --- |
| E2 native 640 vs 1280 | AP-small 0,021894 → 0,042689; recall pada confidence 0,05 dan IoU 0,50 meningkat 0,279819 → 0,363265 | Pasangan train+val asli 1.600 gambar, lima epoch, batch 4, seed 42; satu seed dan anggaran pilot |
| E2 latensi CPU | Median 28,61 → 120,37 ms; p95 39,17 → 146,49 ms, empat thread | Kenaikan resolusi menaikkan biaya inferensi; ini hasil benchmark lokal ketua |
| E3 pilot pada olahan 640 | AP-small kontrol 0,014290 vs tile 0,003391 | Tile 384 dan 620 optimizer steps; belum eksperimen native 1280 dengan tile 640 sesuai revisi |
| E4 SAHI native | AP-small 0,04265 → 0,03899; recall 0,3628 → 0,3732; median CPU 120,0 → 307,7 ms | Ringkasan dari HANDOFF.md, belum disertai CSV/JSON rinci pada commit acuan; protokolnya dicatat terpisah dari E2 |

E2 native 1280 dengan gambar penuh tetap menjadi kandidat sementara. SAHI meningkatkan recall pada pilot E4, tetapi AP-small turun dan median latensi naik. Skor serta anggaran pilot belum cukup untuk menyatakan model final.

## Revisi yang sudah tersedia dalam kode

- Input utama berasal dari notebook Kaggle UTS, dengan pemeriksaan kecocokan ID, split, source ID, dan pasangan hampir kembar.
- E1 baru memakai YOLO26s sesuai starter; hasil historis YOLO26n freeze 10 tetap diberi identitas aslinya.
- E2 memeriksa pasangan native 640/1280 dan manifest geometri sebelum training atau evaluasi.
- E3 menyiapkan tile 640 dari train native 1280, overlap 20%, visibilitas minimal 50%, maksimum satu tile kosong per tiga tile berobjek, serta validation pada gambar utuh.
- E4 menyiapkan inferensi gambar penuh dan SAHI dari checkpoint yang sama, penyelarasan koordinat ke acuan 640, pilihan konfigurasi SAHI, dan confidence operasional dari F1 validation.
- Evaluasi menyediakan TP/FP/FN dan recall per bin ukuran; benchmark baru menyiapkan median/p90 dengan gambar dan jumlah thread yang sama.
- Test dan INT8 tetap dinonaktifkan secara bawaan. Penguncian kandidat, analisis kesalahan, dan ekspor tersedia sebagai langkah lanjutan.

## Verifikasi dan pekerjaan berikutnya

Pemeriksaan lokal menjalankan 42 sel kode dalam mode audit dengan arsip UTS sebenarnya. Uji tambahan mencakup FP duplikat, prediksi kosong, proyeksi koordinat 1280 → 640, serta geometri tile. Kaggle Save & Run All versi 1 berhasil; [status verifikasi](bukti_analisis/verifikasi_kaggle.json) membedakan audit cloud dari training GPU.

1. Pindahkan data native 640/1280, `audit_pasangan_e2.json`, serta checkpoint ketua melalui penyimpanan tim. Data dan checkpoint tetap berada di luar Git.
2. Verifikasi hash dan pasangan input, lalu jalankan E2/E3/E4 versi revisi pada GPU dengan konfigurasi yang tercatat. Simpan hasil baru terpisah dari hasil historis.
3. Lampirkan keluaran rinci E4, bandingkan kandidat pada validation, dan tentukan anggaran training lanjutan.
4. Setelah konfigurasi dikunci, pulihkan native test secara terpisah dan lakukan evaluasi final. Builder native saat ini baru memulihkan train+validation.
5. Cocokkan administrasi laporan dengan pekerjaan nyata anggota, lalu bagikan akses notebook privat kepada pihak kelas yang ditentukan.
