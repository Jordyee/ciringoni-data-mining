# Ciringoni Data Mining

Final project untuk mendeteksi area kecil kerusakan daun jagung akibat Fall Armyworm (FAW) dengan YOLO26n. Dataset memakai satu kelas anotasi, `infected`.

## Hasil akhir pada 178 gambar test

**YOLO26n native 1280, OpenVINO INT8, dengan inferensi gambar penuh.** Model terpilih dilatih 30 epoch dengan batch 4 dan seed 42. Pemilihan resolusi dan varian ekspor memakai 356 gambar validation. Kalibrasi INT8 memakai 320 gambar train. [Protokol dan perbandingan validation](PROTOKOL_FINAL.md)

| COCO AP-small | COCO mAP50–95 | Presisi | Recall | Latensi CPU median / p95 |
| ---: | ---: | ---: | ---: | ---: |
| 0,03019 | 0,03606 | 0,10829 | 0,37964 | 104,21 / 119,41 ms |

Presisi dan recall menggunakan confidence 0,05 dan IoU 0,50. Test dijalankan pada konfigurasi yang sudah dikunci. Skor menunjukkan banyak kotak terlewat bahkan setelah seleksi validation. Proyek mengukur kinerja pada satu koleksi citra dan satu seed. [Lock sebelum test](test-lock.json) · [Metrik lengkap](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_final.csv) · [Notebook M4](Data_Mining_Ciringoni_M1/FP_Data_Mining_M4_Penutupan.ipynb)

[Replikasi Kaggle dan indeks bukti mingguan](kaggle/README.md).

## Mulai bekerja

```powershell
git clone https://github.com/Jordyee/ciringoni-data-mining.git
cd ciringoni-data-mining
```

Baca [HANDOFF.md](HANDOFF.md) dan [protokol penutupan](PROTOKOL_FINAL.md), lalu jalankan notebook dan perintah Python dari direktori berikut:

```powershell
cd Data_Mining_Ciringoni_M1
```

Ambil arsip dataset dan checkpoint dari penyimpanan tim yang aksesnya sudah diberikan; berkas tersebut tidak ikut di repo publik. Siapkan lingkungan sesuai versi di handoff. Dataset acuan dicatat di [notebook Kaggle UTS](https://www.kaggle.com/code/davejordygerungan/data-mining-mid).

## Satu tugas, satu commit

Jalankan Git dari root repo (`cd ..` jika masih di folder proyek). Contoh untuk tugas dokumentasi:

```powershell
git switch -c codex/perbarui-dokumentasi
# Edit README.md dan catat perubahan serta verifikasi di HANDOFF.md.
git diff --check
git add README.md HANDOFF.md
git diff --cached
git commit -m "Update project documentation and handoff"
git push -u origin codex/perbarui-dokumentasi
```

Verifikasi perubahan sebelum commit, periksa isi staged agar aman untuk publik, lalu buka pull request berisi tujuan dan hasil verifikasi. Gunakan [template pull request](.github/pull_request_template.md) sebagai panduan. Untuk tugas lain, pilih hanya berkas yang terkait saat `git add`.
