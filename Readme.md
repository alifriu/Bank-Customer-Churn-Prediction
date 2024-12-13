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

1. Mengidentifikasi fitur-fitur yang berpengaruh untuk modeling machine learning dalam memprediksi keputusan churn pelanggan.
2. Mengembangkan dan menguji model prediktif yang dapat memprediksi pelanggan yang berpotensi churn secara akurat. Model ini akan menggunakan data historis pelanggan dan diuji dengan beberapa algoritma machine learning untuk menentukan model yang paling efektif.

### Solution statements

1. Membangun model prediksi menggunakan tiga algoritma machine learning: Decision Tree, Support Vector Machine (SVM), dan Random Forest.
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

### Exploratory Data Analysis (EDA)

EDA (Exploratory Data Analysis) adalah proses untuk memahami karakteristik data sebelum melakukan pemodelan. Berikut penjelasan singkat mengenai jenis EDA :

1. **Exploring Missing Value dan Duplicate Data**

Pada dataset terdapat missing value pada beberapa kolom.Berikut penjabarannya:
Kolom|jml
----|----
Geography|1
Age|1
HasCrCard|1
IsActiveMember|1

Terdapat juga 2 row yang memiliki nilai duplikat.

2. **Univariate Analysis**
   Analisis ini memeriksa satu variabel pada satu waktu. Tujuannya untuk memahami distribusi, nilai rata-rata, varians, serta mendeteksi outliers.

- Fitur Kategorikal

  <img src='https://github.com/alifriu/Bank-Customer-Churn_Prediction/blob/master/src/uni_cat.png?raw=true'></img>
  **Insight** : Distribusi dari dataset yang ada menunjukan bahwa `Geography` asal dari pelanggan berasal dari 3 negara yaitu France,Spain, dan Germany.
  Pada variabel `Gender`,terlihat bahwa jumlah pelanggan pria lebih dominan dibandingkan dengan pelanggan wanita.

- Fitur Numerik

  <img src='https://github.com/alifriu/Bank-Customer-Churn_Prediction/blob/master/src/uni_churn.png?raw=true'></img>

  **Insight:** Distribusi dari pelanggan yang sudah tidak berlangganan (1) terhadap pelanggan yang masih berlangganan (0) terlihat bahwa masih banyak pelanggan yang masih berlangganan.

  <img src='https://github.com/alifriu/Bank-Customer-Churn_Prediction/blob/master/src/uni_hist.png?raw=true'></img>
  Persebaran data dari tiap variabel pada dataset

  <img src='https://github.com/alifriu/Bank-Customer-Churn_Prediction/blob/master/src/uni_boxplot.png?raw=true'></img>
  Boxplot merupakan visualisasi yang efektif untuk mendeteksi data pencilan (outlier) pada variabel numerik. Dengan menampilkan kuartil data, boxplot memberikan gambaran jelas tentang sebaran data dan mengidentifikasi nilai-nilai ekstrem yang berada di luar jangkauan interkuartil. Titik-titik data yang terletak di luar 'kumis' boxplot umumnya dianggap sebagai outlier.
  Variabel Creditscore dan Age memiliki data outlier yang cukup banyak jika dibandingkan dengan variabel lainnya.

  3. **Bivariate Analysis**
     Analisis ini mengevaluasi hubungan antara dua variabel. Tujuan utamanya adalah untuk menemukan korelasi atau pola antara dua fitur.

  <img src='https://github.com/alifriu/Bank-Customer-Churn_Prediction/blob/master/src/bi_churnGeo.png?raw=true'></img>

  **Insight:** bedasarkan persebaran churn berdasarkan geography,France menjadi negara terbanyak memiliki pelanggan loyal dan churn disaat yang bersamaan.

  <img src='https://github.com/alifriu/Bank-Customer-Churn_Prediction/blob/master/src/bi_churnGen.png?raw=true'></img>

  **Insight:** bedasarkan persebaran churn berdasarkan Gender,Wanita sedikit lebih banyak yang berhenti menggunakan layanan dibandingkan pria.

  <img src='https://github.com/alifriu/Bank-Customer-Churn_Prediction/blob/master/src/bi_density.png?raw=true'></img>

  Density plot merupakan visualisasi yang berguna untuk menggambarkan distribusi dari kepadatan suatu variabel.Berdasarkan data yang ada,kepadatan dari pelanggan yang tidak churn pada masing masing variabel lebih banyak jika dibandingkan dengan pelanggan yang churn.

  4. **Multivariate Analysis**
     Analisis ini melibatkan lebih dari dua variabel sekaligus. Ini bertujuan untuk melihat interaksi kompleks antar variabel dan pola tersembunyi dalam data.

  <img src='https://github.com/alifriu/Bank-Customer-Churn_Prediction/blob/master/src/multi_avgbal.png?raw=true'></img>

  **Insight:** Dari Rata-rata saldo pelangan di tiap Negara,German menjadi negarara yang memiliki rata-rata saldo tertinggi diantara dua negara yang lain.

  untuk mengamati hubungan antara fitur numerik,akan dilakukan visualisasi menggunakan pairplot dan untuk mengobservasi korelasi antara fitur numerik,akan dilakukan visualisasi menggunakan heatmap.

  <img src='https://github.com/alifriu/Bank-Customer-Churn_Prediction/blob/master/src/multi_pairplot.png?raw=true'></img>
  <img src='https://github.com/alifriu/Bank-Customer-Churn_Prediction/blob/master/src/conf_matrix.png?raw=true'></img>

  **Insight:** berdasarkan visualisasi diatas,hubungan korelasi antar variabel bisa dibilang cukup rendah.

## Data Preparation

- **Handling Minssing Value dan Duplicate data**

  Beberapa baris terdapat missing value dan duplicate value di dalamnya,metode yang diterapkan antara lain :

  - Mengisi nilai missing value dengan menggunakan metode ffill.Ini bertujuan untuk mengisi nilai baris yang kosong pada kolom `HasCrCard` dan `IsActiveMember` dengan nilai baris sebelumnya.
  - Variabel `Geography` bertipe data kategorikal sehingga dapat diisi nilainya dengan menggunakan modus yang ada pada kolom.Ini bertujuan untuk memperkecil Variasi data yang ada pada kolom.
  - Dikarenakan variabel `Age` bertipe numerik, dapat dilakukan proses handling missing value dengan menggantinya denga nilai median.Metode ini bertujuan untuk meminimalisir magnitude dari skew dan outlier pada kolom.
  - Drop baris yang memiliki nilai duplikat dari baris lain dan mengecek apakah nilai duplikat masih ada atau tidak

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

   - max_depth : Kedalaman maksimum pohon. `[None,2,3,4]`
   - min_samples_leaf : Jumlah sampel minimum yang diperlukan untuk membagi simpul internal. `[10,20,50]`

   Hyperparameter terbaik setelah tunning : `{'max_depth': None, 'min_samples_leaf': 10}`

2. **Support Vector Machine :** SVM adalah algoritma klasifikasi yang bekerja dengan mencari hyperplane optimal yang memisahkan data ke dalam kelas yang berbeda. Algoritma ini menggunakan support vectors untuk menentukan margin terbaik antara kelas yang memaksimalkan jarak (margin) antara data dari kedua kelas. SVM sangat efektif dalam kasus klasifikasi biner dan dapat bekerja pada data berdimensi tinggi.

   **Kelebihan:**

   - Efektif pada data berdimensi tinggi: SVM bekerja baik pada data dengan banyak fitur.
   - Hasil yang baik pada data dengan margin yang jelas: Jika data terpisah dengan baik, SVM memberikan hasil yang akurat.

   **Kekurangan:**

   - Interpretasi yang sulit: SVM lebih kompleks untuk dipahami dan diinterpretasikan dibandingkan Decision Tree.

   Hyperparameter yang gunakan :

   - C : Parameter regularisasi. Kekuatan regularisasi berbanding terbalik dengan C. `[0.001, 0.01, 0.1, 1]`
   - kernel : Menentukan jenis kernel yang akan digunakan dalam algoritma. `['linear', 'poly', 'rbf', 'sigmoid']`
   - gamma : Koefisien kernel untuk 'rbf', 'poli' dan 'sigmoid'. `['scale', 'auto']`

   Hyperparameter terbaik setelah tunning : `{'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}`

3. **Random Forest Classifier :** Random Forest adalah metode ensemble yang menggabungkan beberapa Decision Tree untuk membuat prediksi yang lebih akurat dan stabil. Algoritma ini membuat banyak pohon keputusan dengan memilih subset data dan fitur secara acak untuk setiap pohon, dan hasil akhir ditentukan dengan voting atau rata-rata dari hasil pohon-pohon tersebut.

   **Kelebihan:**

   - Mengurangi overfitting: Menggunakan banyak pohon yang mengurangi risiko overfitting pada data.

   **Kekurangan:**

   - Waktu komputasi yang lama: Pelatihan Random Forest memerlukan lebih banyak waktu dibandingkan Decision Tree.
   - Memerlukan tuning: Memilih jumlah pohon dan parameter lainnya bisa memerlukan tuning yang cukup kompleks untuk performa optimal.

   Hyperparamter yang digunakan :

   - n_estimators : Jumlah pohon dalam ensemble. `[50,100,200]`
   - max_depth : The maximum depth of the tree. `[None,2,3,4]`
   - min_samples_leaf : The minimum number of samples required to split an internal node. `[10,20,50]`

   Hyperparameter terbaik setelah tunning : `{'max_depth': None, 'min_samples_leaf': 10, 'n_estimators': 200}`

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

   <img src='https://github.com/alifriu/Bank-Customer-Churn_Prediction/blob/master/src/cm_dectree.png?raw=true'></img>

2. **Support Vector Machine**
   | Exited | Precision | Recall | F1-Score |
   |--------|-----------|--------|----------|
   | 0 | 0.79 | 0.81 | 0.80 |
   | 1 | 0.80 | 0.79 | 0.80 |

   <img src='https://github.com/alifriu/Bank-Customer-Churn_Prediction/blob/master/src/cm_svm.png?raw=true'></img>

3. **Random Forest Classifier**
   | Exited | Precision | Recall | F1-Score |
   |--------|-----------|--------|----------|
   | 0 | 0.86 | 0.86 | 0.86 |
   | 1 | 0.86 | 0.86 | 0.86 |

   <img src='https://github.com/alifriu/Bank-Customer-Churn_Prediction/blob/master/src/cm_rf.png?raw=true'></img>

### Perbandingan Accuracy

| Model         | Train | Test |
| ------------- | ----- | ---- |
| Decision Tree | 0.89  | 0.82 |
| SVM           | 0.81  | 0.79 |
| RandomForest  | 0.90  | 0.85 |

<img src='https://github.com/alifriu/Bank-Customer-Churn_Prediction/blob/master/src/acc_tot.png?raw=true'></img>

Setelah dilakukan modeling, didapati bahwa fitur-fitur seperti **CreditScore**, **Geography**, **Gender**, **Age**, **Tenure**, **Balance**, **NumOfProducts**, **HasCrCard**, **IsActiveMember**, dan **EstimatedSalary** memiliki pengaruh signifikan dalam memprediksi churn pelanggan. Setiap fitur ini berkontribusi dalam menentukan apakah pelanggan cenderung untuk berhenti menggunakan layanan atau tetap aktif. Hasil ini sejalan dengan tujuan awal, yaitu mengidentifikasi fitur-fitur yang berpengaruh untuk modeling machine learning dalam memprediksi keputusan churn pelanggan, sehingga memberikan pemahaman yang lebih mendalam kepada bank mengenai karakteristik yang perlu diperhatikan dalam strategi retensi pelanggan.

Dalam evaluasi performa model, **Random Forest** menunjukkan kinerja terbaik dengan akurasi tinggi pada set pengujian (test set) dan F1-Score yang baik untuk kedua kelas, sehingga memenuhi harapan untuk membangun model prediktif yang akurat. Sebaliknya, **Support Vector Machine** memiliki performa terendah dari segi akurasi, sedangkan **Decision Tree** memberikan keseimbangan yang baik antara precision, recall, dan F1-Score pada kedua set (train dan test). Berdasarkan hasil evaluasi, model **Random Forest** dipilih sebagai model terbaik untuk mengidentifikasi pelanggan yang berpotensi churn.

Dengan menggunakan **Random Forest** sebagai model utama, solusi yang diusulkan berhasil memenuhi **Problem Statements** dan **Goals** yang ditetapkan. Random Forest tidak hanya mengidentifikasi faktor utama dalam churn tetapi juga mampu memprediksi pelanggan yang kemungkinan besar akan churn, sehingga berdampak positif pada pemahaman dan strategi bisnis bank. Hasil prediksi ini dapat mendukung bank dalam mengambil langkah proaktif untuk mempertahankan pelanggan yang berpotensi churn, seperti dengan program loyalitas atau penawaran khusus, yang pada akhirnya dapat membantu menurunkan angka churn dan meningkatkan retensi pelanggan.
