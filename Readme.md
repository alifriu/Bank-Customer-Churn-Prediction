# Laporan Proyek Machine Learning - Muhammad Alif Alfattah Riu

## Domain Proyek

Churn pelanggan adalah tantangan utama dalam industri perbankan. Pelanggan yang meninggalkan bank dapat menyebabkan penurunan pendapatan dan peningkatan biaya akuisisi pelanggan baru. Mengingat pentingnya mempertahankan pelanggan, proyek ini bertujuan untuk memprediksi churn menggunakan machine learning.

Masalah ini penting karena melalui prediksi yang akurat, bank dapat mengambil langkah-langkah untuk mencegah churn. Dengan demikian, prediksi churn tidak hanya membantu dalam mempertahankan pelanggan tetapi juga meningkatkan keuntungan dan loyalitas pelanggan.

Beberapa penelitian menunjukkan bahwa biaya mempertahankan pelanggan jauh lebih rendah daripada mendapatkan pelanggan baru. Oleh karena itu, menggunakan model prediktif untuk churn menjadi solusi yang efisien bagi bank untuk menjaga stabilitas bisnis mereka.

## Business Understanding

### Problem Statements

1. Bagaimana cara mengidentifikasi pelanggan yang berpotensi churn berdasarkan data historis pelanggan?
2. Bagaimana cara membangun model prediktif yang dapat memprediksi pelanggan yang berpotensi berhenti menggunakan layanan atau churn?

### Goals

1. Mengidentifikasi karakteristik dan fitur utama yang mempengaruhi kemungkinan pelanggan untuk churn, menggunakan data historis pelanggan. Analisis ini akan memberikan pemahaman yang lebih mendalam kepada bank mengenai faktor-faktor yang memengaruhi keputusan churn.
2. Mengembangkan dan menguji model prediktif yang dapat memprediksi pelanggan yang berpotensi churn secara akurat. Model ini akan menggunakan data historis pelanggan dan diuji dengan beberapa algoritma machine learning untuk menentukan model yang paling efektif.

### Solution statements

1. Membangun model prediksi menggunakan tiga algoritma machine learning: Logistic Regression, Support Vector Machine (SVM), dan Random Forest.
2. Melakukan perbandingan kinerja model dalam memprediksi pelanggan yang kemungkinan besar churn dengan menggunakan metrik evaluasi seperti akurasi, presisi, recall, dan F1-score. Hasil perbandingan ini akan membantu bank memilih algoritma yang paling akurat dalam mengidentifikasi pelanggan yang berpotensi churn.

## Data Understanding

Dataset yang digunakan merupakan dataset yang umum digunakan untuk memprediksi churn pelanggan di industri perbankan .Dataset ini mencakup informasi seperti usia pelanggan, saldo, lama menjadi nasabah, dan status keaktifan. Variabel target adalah Exited yang menunjukkan apakah pelanggan telah churn atau tetap menggunakan layanan bank.[Kaggle:Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction/data).

### Variabel-variabel pada Dataset:

- RowNumber `int64` : Pengenal unik untuk setiap baris dalam dataset. (ex : 1,2,3,dst)
- Customer ID `int64` : Pengenal unik untuk setiap pelanggan. (ex: 15634602)
- Surname `object` : Nama keluarga atau nama belakang pelanggan. (ex: Hargrave, Hill, Onio, dll)
- CreditScore `int64`: Nilai numerik yang mewakili skor kredit pelanggan. (ex : 619, 608, 502, dst)
- Geography `object`: negara tempat tinggal pelanggan. (ex: France, Spain, or Germany)
- Gender `object`: Jenis kelamin pelanggan. (ex: Male or Female).
- Age `int64`: Usia pelanggan.
- Tenure `int64`: Lama Tahunan menjadi nasabah. (ex: 2,1,8,1,dst)
- Balance `float64`: Saldo rekening pelanggan. (ex:83807.86)
- NumOfProducts `int64`: Jumlah produk yang digunakan pelanggan (misal:rekening tabungan, kartu kredit). (ex: 1,2,3,dst)
- HasCrCard `float64`: Kepemilikan kartu kredit (1: ya, 0: tidak).
- IsActiveMember `float64`: Status keaktifan pelanggan (1: aktif, 0: tidak).
- EstimatedSalary `float64`: Gaji perkiraan pelanggan. (ex: 101348.88)
- Exited `int64`: Status churn (1: churn, 0: tetap).

  **Exploratory Data Analysis (EDA)** adalah proses untuk memahami karakteristik data sebelum melakukan pemodelan. Berikut penjelasan singkat mengenai jenis EDA:

  - **Handling Minssing Value dan Duplicate data**

    Beberapa baris terdapat missing value dan duplicate value di dalamnya,metode yang diterapkan antara lain :

    - Mengisi nilai missing value dengan menggunakan metode ffill.Ini bertujuan untuk mengisi nilai baris yang kosong pada kolom `HasCrCard` dan `IsActiveMember` dengan nilai baris sebelumnya.
    - Variabel `Geography` bertipe data kategorikal sehingga dapat diisi nilainya dengan menggunakan modus yang ada pada kolom.Ini bertujuan untuk memperkecil Variasi data yang ada pada kolom.
    - Dikarenakan variabel `Age` bertipe numerik, dapat dilakukan proses handling missing value dengan menggantinya denga nilai median.Metode ini bertujuan untuk meminimalisir magnitude dari skew dan outlier pada kolom.
    - Drop baris yang memiliki nilai duplikat dari baris lain dan mengecek apakah nilai duplikat masih ada atau tidak
      <br>
      <br>

  1. **Univariate Analysis**
     Analisis ini memeriksa satu variabel pada satu waktu. Tujuannya untuk memahami distribusi, nilai rata-rata, varians, serta mendeteksi outliers.

  - Fitur Kategorikal

    <img src='https://private-user-images.githubusercontent.com/119936884/381028526-40052069-ee75-4179-a0ca-d610708b94d3.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAxODg4NjgsIm5iZiI6MTczMDE4ODU2OCwicGF0aCI6Ii8xMTk5MzY4ODQvMzgxMDI4NTI2LTQwMDUyMDY5LWVlNzUtNDE3OS1hMGNhLWQ2MTA3MDhiOTRkMy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAyOVQwNzU2MDhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zOTFmYTM3ZjVhNjJiODdkN2E3ZWZiMDA0ODY1MjlkMjhhZGU5NzZjOGE3NTgwNjEzMmQ0NDY0NDM5NjZlYTdjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.sMEfuhjwVMh3aHNjYJH7eQLpxv9tSF2oDhxhmjheYcY'></img>
    **Insight** : Distribusi dari dataset yang ada menunjukan bahwa `Geography` asal dari pelanggan berasal dari 3 negara yaitu France,Spain, dan Germany.
    Pada variabel `Gender`,terlihat bahwa jumlah pelanggan pria lebih dominan dibandingkan dengan pelanggan wanita.

  - Fitur Numerik

    <img src='https://private-user-images.githubusercontent.com/119936884/381028532-c43e5e66-633c-4042-825f-5bbbf15a9fa2.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAxODg4NjgsIm5iZiI6MTczMDE4ODU2OCwicGF0aCI6Ii8xMTk5MzY4ODQvMzgxMDI4NTMyLWM0M2U1ZTY2LTYzM2MtNDA0Mi04MjVmLTViYmJmMTVhOWZhMi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAyOVQwNzU2MDhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jN2IzMzM1YjkyZGY2MDBmY2I5NjM3MGRjODNmOTk5OGJjOWVjZDkwMTEwMTdiYjQ0NDk2NzdhYzE0NjlmNTliJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.V_L0myxnj6rcba9Wt2nm-QzFClBUBWm1IO5pbrhfmqk'></img>

    **Insight:** Distribusi dari pelanggan yang sudah tidak berlangganan (1) terhadap pelanggan yang masih berlangganan (0) terlihat bahwa masih banyak pelanggan yang masih berlangganan.

    <img src='https://private-user-images.githubusercontent.com/119936884/381028464-a237575f-b22a-4d3e-ad3d-90e96792f46f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAxODg4NjgsIm5iZiI6MTczMDE4ODU2OCwicGF0aCI6Ii8xMTk5MzY4ODQvMzgxMDI4NDY0LWEyMzc1NzVmLWIyMmEtNGQzZS1hZDNkLTkwZTk2NzkyZjQ2Zi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAyOVQwNzU2MDhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kOWJlZDBjYmE4OTQyNDI1YzZiMWE2ZWMxNTEwZWFkMDMyYzEyOGI1OWI0Mzg5NzFkZTNhOTk1OWY4MDQ2NGJhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.l48MPJ9lhCYNkVO2fyRTjpbs3MSPOLikQSjw4XElgdM'></img>
    Persebaran data dari tiap variabel pada dataset

    <img src='https://private-user-images.githubusercontent.com/119936884/381028520-2bb8c3e8-7e14-4610-9d9f-60155f8cae3d.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAxODg4NjgsIm5iZiI6MTczMDE4ODU2OCwicGF0aCI6Ii8xMTk5MzY4ODQvMzgxMDI4NTIwLTJiYjhjM2U4LTdlMTQtNDYxMC05ZDlmLTYwMTU1ZjhjYWUzZC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAyOVQwNzU2MDhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jZTQ2NjllYWExZmRmYTlhMGZkMDI3N2QzOWMxNDk0MWNiNjRjM2YwZjA1YzI1OGI2OGJjMjI4MGZlMzE0NGExJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.2Rc2ege9R99e5r7DYunjjho-8BMltTPsJb4Las1ed5Q'></img>
    Boxplot merupakan visualisasi yang efektif untuk mendeteksi data pencilan (outlier) pada variabel numerik. Dengan menampilkan kuartil data, boxplot memberikan gambaran jelas tentang sebaran data dan mengidentifikasi nilai-nilai ekstrem yang berada di luar jangkauan interkuartil. Titik-titik data yang terletak di luar 'kumis' boxplot umumnya dianggap sebagai outlier.
    Variabel Creditscore dan Age memiliki data outlier yang cukup banyak jika dibandingkan dengan variabel lainnya.

    2. **Bivariate Analysis**
       Analisis ini mengevaluasi hubungan antara dua variabel. Tujuan utamanya adalah untuk menemukan korelasi atau pola antara dua fitur.

    <img src='https://private-user-images.githubusercontent.com/119936884/381028476-b4554b12-359d-4f4b-9a30-0a9bc6366153.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAxODg4NjgsIm5iZiI6MTczMDE4ODU2OCwicGF0aCI6Ii8xMTk5MzY4ODQvMzgxMDI4NDc2LWI0NTU0YjEyLTM1OWQtNGY0Yi05YTMwLTBhOWJjNjM2NjE1My5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAyOVQwNzU2MDhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03YjhkYTA2ZjZmYTAzNGQ4MzBlYzVjYWJhMTk3ZDIzOWVhZTJjNzU4Yjc5ZThiMzI1YjBiMWRmYTc0NTU3MTg2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.Xzxp4rPALNInS8I1fxe1ujYs88_EgWM8cJ4_mShiR_Q'></img>

    **Insight:** bedasarkan persebaran churn berdasarkan geography,France menjadi negara terbanyak memiliki pelanggan loyal dan churn disaat yang bersamaan.

    <img src='https://private-user-images.githubusercontent.com/119936884/381028470-18ffb97c-45a0-4f08-8f42-01de5b30356e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAxODg4NjgsIm5iZiI6MTczMDE4ODU2OCwicGF0aCI6Ii8xMTk5MzY4ODQvMzgxMDI4NDcwLTE4ZmZiOTdjLTQ1YTAtNGYwOC04ZjQyLTAxZGU1YjMwMzU2ZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAyOVQwNzU2MDhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lMjcwYjVhM2E0MmE3ZjFiMjg0OWM5MjI1OTcwNzJhODA0M2FmOWJmZmJhN2M0Y2ExODk4MTgzYzQyNDM4ODYzJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.vE2NbE01ueZs6ZhZ6-wnls6fBhGbPz5hyh9oBm45n2I'></img>

    **Insight:** bedasarkan persebaran churn berdasarkan Gender,Wanita sedikit lebih banyak yang berhenti menggunakan layanan dibandingkan pria.

    <img src='https://private-user-images.githubusercontent.com/119936884/381028482-dd9b3585-f6b3-41e8-8509-68939ba11584.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAxODg4NjgsIm5iZiI6MTczMDE4ODU2OCwicGF0aCI6Ii8xMTk5MzY4ODQvMzgxMDI4NDgyLWRkOWIzNTg1LWY2YjMtNDFlOC04NTA5LTY4OTM5YmExMTU4NC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAyOVQwNzU2MDhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04NTc5MGVlNzUyYzljNmVjNjI1NTZlYWYyYWNiOTcwZmQ3NDlkMTA5ZjBkYWY0MjE3MjNhNWMxNzE3OWVjZWI1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.7pCx8Nosk8XU5QAnevuNjttxvy3ntasZ9ALlXFGAu1c'></img>

    Density plot merupakan visualisasi yang berguna untuk menggambarkan distribusi dari kepadatan suatu variabel.Berdasarkan data yang ada,kepadatan dari pelanggan yang tidak churn pada masing masing variabel lebih banyak jika dibandingkan dengan pelanggan yang churn.

    3. **Multivariate Analysis**
       Analisis ini melibatkan lebih dari dua variabel sekaligus. Ini bertujuan untuk melihat interaksi kompleks antar variabel dan pola tersembunyi dalam data.

    <img src='https://private-user-images.githubusercontent.com/119936884/381028504-006e7f93-b18e-4452-a2d0-e000edbf91e8.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAxODg4NjgsIm5iZiI6MTczMDE4ODU2OCwicGF0aCI6Ii8xMTk5MzY4ODQvMzgxMDI4NTA0LTAwNmU3ZjkzLWIxOGUtNDQ1Mi1hMmQwLWUwMDBlZGJmOTFlOC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAyOVQwNzU2MDhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03NTI3ZjJjMTA5MDU2M2M1YTRmZGUwMjQ1YzdjMTE0ZTQ5OGFlZDhmNWNiYzBiMjRlMmRhNDQ0Mzc5OGUzOGU2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.f1J2Vx4zWyBwAuXyoK9DhDtUvZEMRxm5_l2u0hR6IF4'></img>

    **Insight:** Dari Rata-rata saldo pelangan di tiap Negara,German menjadi negarara yang memiliki rata-rata saldo tertinggi diantara dua negara yang lain.

    untuk mengamati hubungan antara fitur numerik,akan dilakukan visualisasi menggunakan pairplot dan untuk mengobservasi korelasi antara fitur numerik,akan dilakukan visualisasi menggunakan heatmap.

    <img src='https://private-user-images.githubusercontent.com/119936884/381028507-9186217a-9464-42f3-b1c8-4289f244fe88.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAxODg1MjksIm5iZiI6MTczMDE4ODIyOSwicGF0aCI6Ii8xMTk5MzY4ODQvMzgxMDI4NTA3LTkxODYyMTdhLTk0NjQtNDJmMy1iMWM4LTQyODlmMjQ0ZmU4OC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAyOVQwNzUwMjlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xMTg2YjBiY2IxYzVjZmEwMWU5Y2I1MTU0YmY3MmY2ZTk5NDhiMmE1ODE2NzA0NmIxNzZkNWMyNTgwMWJmMGViJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.wrekUaCbLq56tEigfdgpvtAZnxks9FALoTsNzU9II9g'></img>
    <img src='https://private-user-images.githubusercontent.com/119936884/381028499-f422fd58-d0e0-4c30-8efc-66c5fa13c07f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAxODg1MjksIm5iZiI6MTczMDE4ODIyOSwicGF0aCI6Ii8xMTk5MzY4ODQvMzgxMDI4NDk5LWY0MjJmZDU4LWQwZTAtNGMzMC04ZWZjLTY2YzVmYTEzYzA3Zi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAyOVQwNzUwMjlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zZjJjYmJlZjRjZTRiMmI1OTE3ODFlZjEzYmYxOTBkZDRiZTU5ODllZGI2Y2U4Y2ViY2I5MDk3MjRmMTU4MDkwJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.3fLFSzsoiesE8W5fvi1QggrEW5sAhrfqgq8N4konTTo'></img>

    **Insight:** berdasarkan visualisasi diatas,hubungan korelasi antar variabel bisa dibilang cukup rendah.

## Data Preparation

- **Enkoding Fitur Kategorikal**:
  Untuk memungkinkan algoritma machine learning memproses kolom bertipe data kategorikal seperti 'Geography' dan 'Gender', kita perlu mengubahnya menjadi format numerik. Proses ini disebut encoding. Setiap kategori unik dalam variabel tersebut akan diubah menjadi vektor biner, di mana hanya satu nilai yang akan bernilai 1 (true) dan sisanya 0. Dengan demikian, informasi dari variabel kategorikal dapat dimasukkan secara efektif ke dalam model.

- **Resample untuk Data Kelas Minoritas**:
  Teknik resample data adalah suatu metode yang digunakan dalam pra-pemrosesan data, khususnya dalam Machine Learning, untuk mengatasi ketidakseimbangan kelas (imbalanced class) dalam dataset. Ketidakseimbangan kelas terjadi ketika jumlah data pada satu kelas jauh lebih banyak daripada kelas lainnya. Kondisi ini dapat menyebabkan model Machine Learning menjadi bias terhadap kelas mayoritas dan mengabaikan kelas minoritas.

    Dalam Dataset yang digunakan,Kolom Exited untuk kelas
    Sebelum Oversampling | Jumlah | Sedudah OverSampling | Jumlah
    --------|---------|-------|-----|
    Tidak Exited/Churn (0) | 7963 | Tidak Exited/Churn (0) | 7963
    Exited/Churn (1) | 2037 | Exited/Churn (1) | 7963

- **Split Data Training dan Data Testing**:
  Tujuan utama dari kasus ini adalah memprediksi apakah pelanggan akan churn atau tidak (variabel 'Exited'). Untuk mencapai tujuan ini, kita akan membagi data menjadi dua bagian: 80% untuk melatih model (train set) dan 20% untuk menguji performanya (test set). Train set digunakan untuk mengajarkan model mengenali pola yang mengindikasikan churn, sedangkan test set digunakan untuk mengukur seberapa akurat model dalam memprediksi pelanggan baru yang akan churn.

- **Standardisasi Data**:
  Standarisasi data menggunakan StandardScaler bertujuan untuk menormalkan distribusi setiap fitur agar memiliki rata-rata sama dengan nol dan penyimpangan baku sama dengan satu. Teknik ini memastikan bahwa semua fitur berkontribusi secara setara dalam model, terlepas dari skala pengukuran awal.

## Modeling

Model yang digunakan dalam proses pemodelan antara lain Decision Tree Classifier,Support Vector Machine, dan Random Forest Classifier dan Hyperparatemer tunning menggunakan GridSearchCV.

1. **Decission Tree Classifier :** Decision Tree adalah algoritma machine learning yang menggunakan struktur seperti pohon untuk membuat keputusan berdasarkan data input. Algoritma ini membagi data berdasarkan fitur-fitur tertentu, dan tiap cabang merepresentasikan keputusan yang diambil. Pada setiap node, data dipecah menggunakan aturan tertentu untuk mengurangi impurity sampai mencapai leaf node, yang merupakan hasil akhir atau kelas prediksi.

   **Kelebihan:**

   - Mudah diinterpretasi: Struktur pohon memungkinkan interpretasi yang jelas tentang proses pengambilan keputusan.
   - Dapat menangani data kategorikal dan numerik: Fleksibel untuk digunakan pada berbagai jenis data.

   **Kekurangan:**

   - Overfitting: Decision Tree rentan terhadap overfitting, terutama jika pohonnya terlalu dalam.

   Hyperparameter yang gunakan :

   - max_depth : Kedalaman maksimum pohon. [None,2,3,4]
   - min_samples_leaf : Jumlah sampel minimum yang diperlukan untuk membagi simpul internal. [10,20,50]

2. **Support Vector Machine :** SVM adalah algoritma klasifikasi yang bekerja dengan mencari hyperplane optimal yang memisahkan data ke dalam kelas yang berbeda. Algoritma ini menggunakan support vectors untuk menentukan margin terbaik antara kelas yang memaksimalkan jarak (margin) antara data dari kedua kelas. SVM sangat efektif dalam kasus klasifikasi biner dan dapat bekerja pada data berdimensi tinggi.

   **Kelebihan:**

   - Efektif pada data berdimensi tinggi: SVM bekerja baik pada data dengan banyak fitur.
   - Hasil yang baik pada data dengan margin yang jelas: Jika data terpisah dengan baik, SVM memberikan hasil yang akurat.

   **Kekurangan:**

   - Interpretasi yang sulit: SVM lebih kompleks untuk dipahami dan diinterpretasikan dibandingkan Decision Tree.

   Hyperparameter yang gunakan :

   - C : Parameter regularisasi. Kekuatan regularisasi berbanding terbalik dengan C. [0.001, 0.01, 0.1, 1]
   - kernel : Menentukan jenis kernel yang akan digunakan dalam algoritma. ['linear', 'poly', 'rbf', 'sigmoid']
   - gamma : Koefisien kernel untuk 'rbf', 'poli' dan 'sigmoid'. ['scale', 'auto']

3. **Random Forest Classifier :** Random Forest adalah metode ensemble yang menggabungkan beberapa Decision Tree untuk membuat prediksi yang lebih akurat dan stabil. Algoritma ini membuat banyak pohon keputusan dengan memilih subset data dan fitur secara acak untuk setiap pohon, dan hasil akhir ditentukan dengan voting atau rata-rata dari hasil pohon-pohon tersebut.

   **Kelebihan:**

   - Mengurangi overfitting: Menggunakan banyak pohon yang mengurangi risiko overfitting pada data.

   **Kekurangan:**

   - Waktu komputasi yang lama: Pelatihan Random Forest memerlukan lebih banyak waktu dibandingkan Decision Tree.
   - Memerlukan tuning: Memilih jumlah pohon dan parameter lainnya bisa memerlukan tuning yang cukup kompleks untuk performa optimal.

   Hyperparamter yang digunakan :

   - n_estimators : Jumlah pohon dalam ensemble. [50,100,200]
   - max_depth : The maximum depth of the tree. [None,2,3,4]
   - min_samples_leaf : The minimum number of samples required to split an internal node. [10,20,50]

## Evaluation

Metrik evaluasi yang digunakan pada kasus ini adalah:

- **Akurasi**: Akurasi mengukur proporsi prediksi yang benar terhadap keseluruhan prediksi. Ini adalah metrik yang paling intuitif, tetapi bisa kurang informatif pada dataset yang tidak seimbang.

  **Akurasi**: $$\frac{TP + TN}{TP + TN + FP + FN}$$

- **Precission**: Presisi mengukur proporsi prediksi positif yang benar-benar positif. Ini penting ketika biaya dari prediksi positif yang salah (false positive) tinggi.

  **Precision**: $$\frac{TP}{TP + FP}$$

- **Recall**: Recall mengukur proporsi data positif yang benar-benar diidentifikasi sebagai positif oleh model. Metrik ini penting jika biaya dari prediksi negatif yang salah (false negative) tinggi.

  **Recall**: $$\frac{TP}{TP + FN}$$

- **F1 Score**: F1 Score adalah rata-rata harmonik dari presisi dan recall. Metrik ini memberikan gambaran seimbang ketika kedua metrik penting, serta menyeimbangkan antara false positive dan false negative.

  **F1 Score**: $$2 \times \frac{Precision \times Recall}{Precision + Recall}$$

Di sini, **TP** adalah True Positive, **TN** adalah True Negative, **FP** adalah False Positive, dan **FN** adalah False Negative.

1. **Decision Tree Classifier**
   | Exited | Precision | Recall | F1-Score |
   |--------|-----------|--------|----------|
   | 0 | 0.84 | 0.81 | 0.83 |
   | 1 | 0.82 | 0.85 | 0.83 |

   <img src='https://private-user-images.githubusercontent.com/119936884/381028488-8572c851-9155-49ef-a1c1-b70bce5abbb3.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAxODg1MjksIm5iZiI6MTczMDE4ODIyOSwicGF0aCI6Ii8xMTk5MzY4ODQvMzgxMDI4NDg4LTg1NzJjODUxLTkxNTUtNDllZi1hMWMxLWI3MGJjZTVhYmJiMy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAyOVQwNzUwMjlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zM2RkOTMyOGJkODg1MWExMzU5OWQ4ZjhkZTU1NGEwYjJjY2ExNmJjZmYyYTM4MDY2MTc1YTcyYjQ3MGZlOTUyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.fUDcJ8WSPDQcvUjYsGXHn8kneSWm089FuXcz5C2KsCU'></img>

2. **Support Vector Machine**
   | Exited | Precision | Recall | F1-Score |
   |--------|-----------|--------|----------|
   | 0 | 0.79 | 0.81 | 0.80 |
   | 1 | 0.80 | 0.79 | 0.80 |

   <img src='https://private-user-images.githubusercontent.com/119936884/381028494-274d2cf6-2ef2-4cac-907e-ba5bcefcafa0.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAxODg1MjksIm5iZiI6MTczMDE4ODIyOSwicGF0aCI6Ii8xMTk5MzY4ODQvMzgxMDI4NDk0LTI3NGQyY2Y2LTJlZjItNGNhYy05MDdlLWJhNWJjZWZjYWZhMC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAyOVQwNzUwMjlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1iMzRiOWNjYTY1ZjkwZjQzYWRmYjU4YWE0NTVlZmE4M2NhN2RiYTliZDFiODE5ODY5MzAzYmNiMDExNWIxNTNkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.4l-pa3GseKFglCbY_eHQQd5d6gs7wCqdw8jUYhYIZe0'></img>

3. **Random Forest Classifier**
   | Exited | Precision | Recall | F1-Score |
   |--------|-----------|--------|----------|
   | 0 | 0.86 | 0.86 | 0.86 |
   | 1 | 0.86 | 0.86 | 0.86 |

   <img src='https://private-user-images.githubusercontent.com/119936884/381028492-ecf0b2c7-54da-4680-8ffc-af0d2a9edd17.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAxODg1MjksIm5iZiI6MTczMDE4ODIyOSwicGF0aCI6Ii8xMTk5MzY4ODQvMzgxMDI4NDkyLWVjZjBiMmM3LTU0ZGEtNDY4MC04ZmZjLWFmMGQyYTllZGQxNy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAyOVQwNzUwMjlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0wZjA2Y2FiNTQ4N2JjM2VkN2IxOTdjYWFjODQzMDczMzdlYWI2ZjE4YWJlZmJlMzA0ZDVmYzI4ZDg2NzNhMTJhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.iP4uZE6f55hVtVVpPn5Tx2c5e1eciT8utW7W2pDB-Q8'></img>

### Perbandingan Accuracy

| Model         | Train | Test |
| ------------- | ----- | ---- |
| Decision Tree | 0.89  | 0.82 |
| SVM           | 0.81  | 0.79 |
| RandomForest  | 0.90  | 0.85 |

<img src='https://private-user-images.githubusercontent.com/119936884/381028469-5c017e74-9da6-44d5-9a93-bca2d7f796c3.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAxODg1MjksIm5iZiI6MTczMDE4ODIyOSwicGF0aCI6Ii8xMTk5MzY4ODQvMzgxMDI4NDY5LTVjMDE3ZTc0LTlkYTYtNDRkNS05YTkzLWJjYTJkN2Y3OTZjMy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQxMDI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MTAyOVQwNzUwMjlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0wYjdlZjg0ZjdmZmE3NjYwZmE2M2MyYTkzZDdmNWFiMzZiYWNjNzkwZTJjMmM3ZDNiMGIyNGU0NGM2NzVhZTg5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.OvuYY3FtC6AfZVe_YLkdo3So6AQvC-AUf56rMqfZegc'></img>

Secara keseluruhan, Random Forest menunjukkan performa terbaik dalam akurasi pada set test dan memiliki F1-Score tinggi untuk kedua kelas.Sementara itu, Support Vector Machine memiliki performa terendah dari segi akurasi.Decision Tree memberikan keseimbangan yang cukup baik antara precision, recall, dan F1-Score dengan akurasi yang cukup baik di set train dan set test.

Berdasarkan model yang dipilih,Random Forest merupakan model Random Forest yang terbaik dalam mengidentifikasi pelanggan yang berpotensi churn.
