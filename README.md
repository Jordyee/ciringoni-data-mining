# Ciringoni Data Mining

**Mulai M2 di sini.** [Handoff program, notebook Kaggle, dan prompt untuk Codex penerus](HANDOFF_M2.md).

Final project untuk mendeteksi area kecil kerusakan daun jagung akibat Fall Armyworm (FAW) dengan YOLO26n. Dataset memakai satu kelas anotasi, `infected`.

## Ringkasan pemeriksaan 30–60 detik

**Kandidat sementara adalah YOLO26n E2 native 1280 dengan inferensi gambar penuh.** Pasangan 640/1280 memakai 1.600 citra asli train+validation yang dipulihkan melalui MD5 dan audit geometri. Keduanya dilatih lima epoch, batch 4, seed 42, lalu dievaluasi pada 356 gambar validation yang sama.

| Metrik validation E2 | Native 640 | Native 1280 |
| --- | ---: | ---: |
| AP-small | 0,0219 | 0,0427 |
| Ultralytics mAP50–95 | 0,0199 | 0,0326 |
| Recall (confidence 0,05; IoU 0,50) | 0,2798 | 0,3633 |
| CPU median / p95 (ms, 4 thread) | 28,61 / 39,17 | 120,37 / 146,49 |

[Angka lengkap E2](Data_Mining_Ciringoni_M1/hasil_m2/evaluasi/eksperimen_e2_val.csv). Kenaikan deteksi dibayar latensi CPU lebih tinggi. E4 SAHI dengan bobot E2 yang sama menurunkan AP-small 0,04265→0,03899; median CPU naik 120,0→307,7 ms pada benchmark terpisah. E3 pada data olahan 640, E5, E6, dan pilot E7 FP32/INT8 kontrol 640 sudah selesai. Skor masih rendah dan anggaran baru lima epoch; ekspor kandidat final serta evaluasi 178 gambar test yang belum digunakan tetap tertunda.

## Mulai bekerja

```powershell
git clone https://github.com/Jordyee/ciringoni-data-mining.git
cd ciringoni-data-mining
```

Baca [HANDOFF.md](HANDOFF.md), lalu jalankan notebook dan perintah Python dari direktori berikut:

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
