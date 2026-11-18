# Panduan Kaggle Ciringoni M2

Folder ini memuat notebook revisi berdasarkan starter dosen, Project.pdf serta repo ketua pada commit e376ec2. Periode progres sampai 18 November 2026, dengan verifikasi Kaggle pada 19 November 2026. [Ringkasan progres publik](PROGRESS_M2.md) merangkum hasil dan keterbatasannya; laporan Word/PDF beserta foto tetap disimpan dalam paket internal.

## Notebook yang sudah tersimpan

- [Buka notebook Final Project M2 di Kaggle](https://www.kaggle.com/code/kaawoanprayer/fp-data-mining-ciringoni-m2).
- Versi 1, **Save & Run All**, **Successful dalam 39,1 detik** pada 19 November 2026. Notebook masih privat pada akun Prayer.
- [Lihat log versi yang diverifikasi](https://www.kaggle.com/code/kaawoanprayer/fp-data-mining-ciringoni-m2/log?scriptVersionId=371146508).
- Source Notebook yang terpasang adalah [Data-Mining-MID](https://www.kaggle.com/code/davejordygerungan/data-mining-mid), melalui **Add Input → Notebook**.
- Versi ini menjalankan audit dengan semua `LATIH=False`. Riwayat hasil GPU berasal dari repo; training GPU versi revisi belum dijalankan.

## Berkas yang dipakai

- [FP_DM_Ciringoni_Kaggle_M2.ipynb](FP_DM_Ciringoni_Kaggle_M2.ipynb) untuk Kaggle. [Notebook Colab](FP_DM_Ciringoni_Colab_M2.ipynb) memiliki alur identik dengan deteksi direktori Colab.
- [bukti_analisis](bukti_analisis/) berisi ringkasan pemeriksaan dataset UTS, metrik historis, serta log dan status verifikasi Kaggle. Dataset, bobot, dan tabel per gambar tidak ikut di folder publik ini.
- `ProgressReport_FP_DM_Ciringoni_M2.docx` dan PDF dengan nama sama tersedia dalam paket internal untuk penyuntingan dan pengumpulan progres.

## Impor dan jalankan audit awal

1. Untuk melanjutkan versi yang sudah ada, buka tautan Kaggle di atas melalui akun Prayer lalu pilih Edit. Untuk membuat salinan sendiri, buat New Notebook, kemudian File → Import Notebook dan pilih FP_DM_Ciringoni_Kaggle_M2.ipynb.
2. Gunakan Add Input → Notebook, tempel https://www.kaggle.com/code/davejordygerungan/data-mining-mid lalu pilih output versi UTS yang dipakai kelompok. Pastikan UTS_DM_kelompok_ciringoni_hasil.zip muncul dalam input. Jalankan Bagian1–5. Notebook menyalin dataset ke direktori kerja, mencocokkan pembagian_split.csv, mengaudit source_id dan pasangan hampir kembar, lalu menguji COCOeval dengan prediksi sempurna.
3. Cocokkan ringkasan train1.244/10.540 kotak, val356/2.205, test178/1.159. Total1.778 gambar dan13.904 kotak. Source_id lintas split dan pasangan hampir kembar lintas split harus0.
4. Bagian5 menampilkan CSV historis repo. Status itu merujuk pada eksperimen lokal ketua. Setiap hasil eksekusi baru masuk eksperimen.csv dengan status dihitung pada eksekusi ini.

## Siapkan E2 resolusi tinggi

Notebook Final Project dibuat terpisah dengan sumber output UTS melalui tautan Kaggle di atas. Pada commit acuan e376ec2, repo hanya mencantumkan notebook UTS. Tautan Final Project M2 kini dicatat pada README root dan panduan ini; akses Kaggle tetap privat.

1. Minta ketua memberikan folder native berisi 640/, 1280/ dan audit_pasangan_e2.json yang ready=true. Berkas ini berada di luar GitHub menurut HANDOFF. Pastikan bobot E2 juga ikut dipindahkan.
2. Unggah folder/paket tersebut ke Kaggle Dataset privat dan tambahkan melalui Add Input. Isi INPUT_NATIVE dengan path folder atau ZIP. Notebook akan memeriksa ID serta label train/val, lalu menggunakan manifest untuk memetakan kotak1280 kembali ke acuan640.
3. Alternatif sesuai dosen adalah menjalankan ulang notebook UTS dari citra sumber dengan IMG_SIZE=1280, SEED, MANUAL_EXCLUDE, PILIH_VERSI dan alias source_id yang sama. Gunakan nama kelompok berakhiran _1280. Tambahkan arsip tersebut dan set KONFIRMASI_NATIVE=True setelah memastikan asal sumber serta split. Untuk kontrol E2 native640, gunakan paket pasangan ketua.
4. Untuk memuat hasil ketua, isi CHECKPOINTS dengan ID E2 dan E2_control_native640. Pilih EVALUASI=['E2_control_native640','E2'] dan biarkan LATIH keduanya False. Untuk pelatihan baru, pilih LATIH keduanya True. Gunakan GPU dan budget yang sama pada kedua sisi.
5. Dua SHA-256 checkpoint ketua tercantum dalam HANDOFF.md. E2 native1280 adalah0bd467aef478ebd26555074030ff653c52cf05ad5380fcbe66057b3c995f264e. Kontrol native640 adalah56568daf2c930033058a3b0468f1357069b1110ce2a1e07e705405547f46d34f.

## Jalankan E3 tiling native

1. Gunakan INPUT_NATIVE yang sama. Tambahkan E3 pada EVALUASI dan pilih LATIH['E3']=True untuk pelatihan pertama. Set EKSPOR_TILE=True bila hanya ingin memeriksa ekspor.
2. Bagian8 membuat tile640 hanya dari train1280, overlap20%, visibilitas kotak minimal50%, dan maksimum satu tile kosong per tiga tile berobjek. Enam contoh tile ditampilkan agar geometri bisa diperiksa.
3. Evaluasi E3a memakai val utuh1280. Untuk E3b, atur SAHI_MODEL='E3' dan tambahkan E4 pada EVALUASI. Simpan checkpoint E3 ke Kaggle Dataset setelah training selesai.
4. Hasil E3 lokal repo adalah pilot tile384 dari olahan640 dengan anggaran620 optimizer steps. Gunakan ID dan deskripsi baru untuk E3 native agar hasil dapat ditelusuri.

## Jalankan E4 SAHI

1. Pilih SAHI_MODEL='E2' untuk bobot kandidat ketua atau E3 untuk bobot tiling baru. Isi CHECKPOINTS bila model belum dilatih pada sesi ini.
2. Tambahkan E4 pada EVALUASI. SAHI_PILIH=[0] menjalankan tile640/overlap20% tanpa prediksi penuh. Pilih [0,1,2] untuk tiga konfigurasi yang disediakan.
3. Notebook mengevaluasi gambar penuh dan SAHI pada GT yang sama. Periksa AP_small, recall per bin, FP, dan confidence operasional dari val.
4. Aktifkan UKUR_CPU=True untuk benchmark baru. Jalankan semua model pada jenis CPU dan jumlah thread yang sama. Hasilnya median/p90, batch1, gambar yang sama,10 warmup dan100 pengulangan. Benchmark tersebut memakai wall clock sesudah decoding sampai penggabungan prediksi.

## Simpan versi progres di Kaggle

1. Setelah eksperimen selesai, unduh folder run, hasil CSV dan checkpoint. Tambahkan bobot yang sudah dilatih sebagai input privat.
2. Isi CHECKPOINTS untuk semua run yang ingin dievaluasi, lalu set seluruh LATIH menjadi False. EVALUASI berisi run yang hendak diperiksa ulang.
3. Pilih Save Version → Save & Run All. Tunggu status berhasil. Catat tautan notebook serta nomor versi dalam laporan. Bagikan ke dosen lewat Share → Add collaborator sesuai instruksi kelas.
4. status_eksekusi.json menjelaskan run yang dihitung pada versi tersebut. Arsip FP_DM_Ciringoni_hasil.zip tersedia di Output. Dalam mode audit awal, arsip hanya berisi bukti progres yang dihasilkan saat itu.

## Bagian berikutnya dan pengumpulan

E1 pada starter dosen memakai YOLO26s. Hasil E1 M1 ketua memakai YOLO26n freeze10. Notebook menyediakan E1 YOLO26s dengan ID baru dan tabel historis M1 tetap memuat deskripsi aslinya.

Bagian10–15 menyediakan alur P2, augmentasi, penguncian kandidat, analisis error dan ekspor. JALANKAN_TEST serta JALANKAN_INT8 awal False. Untuk test native1280, dataset harus memuat178 ID test yang cocok. Builder E2 ketua saat ini hanya memulihkan train+val. Label kelas/acuan untuk analisis harus berada pada koordinat640 sebelum dibandingkan.

Tanggal/jam foto, jam kerja M2, nomor kelompok numerik, persentase aktual dan paraf belum tersedia pada bukti. Laporan memakai nama Ciringoni dan usulan20% per anggota. Ketua perlu memeriksa bagian administrasi ini sebelum mengesahkan laporan. Nama kelompok08 pada contoh tidak dipakai sebagai identitas Ciringoni.

## Pemeriksaan yang telah dilakukan

42 sel kode notebook telah dijalankan lokal dalam mode audit dengan arsip UTS sebenarnya. Pemeriksaan mencakup pencocokan split, label, source_id, pasangan hampir kembar, profil ukuran serta uji prediksi sempurna COCOeval. Uji tambahan memeriksa FP duplikat, proyeksi koordinat1280→640 dan geometri tile. Sel GPU belum dieksekusi di lingkungan ini. Notebook Kaggle telah disimpan pada akun Prayer melalui Save & Run All versi1 dan berhasil dalam39,1detik pada19November2026. URL https://www.kaggle.com/code/kaawoanprayer/fp-data-mining-ciringoni-m2 . Akses notebook masih privat. Mode audit berjalan dengan source Notebook Data-Mining-MID. Training GPU serta input native masih menunggu kelanjutan tim.
