# DATASET 
**CUHK Face Sketch Database (CUFS)** digunakan untuk penelitian mengenai sintesis sketsa wajah dan pengenalan sketsa wajah.
 
Database ini mencakup 188 wajah dari database mahasiswa Chinese University of Hong Kong (CUHK), 123 wajah dari database AR, dan 295 wajah dari database XM2VTS. Total terdapat 606 wajah. Untuk setiap wajah, tersedia sebuah sketsa yang digambar oleh seorang seniman berdasarkan foto yang diambil dengan pose frontal, di bawah pencahayaan normal, dan dengan ekspresi netral. Di bawah ini disediakan beberapa pasangan (foto dan sketsa) yang disediakan oleh penulis masing-masing dari ARFACE, database Mahasiswa CUHK, dan XM2VTS.

**#Arsitektur Autoencoder**
Terdiri dari 2 bagian utama : 
- Encoder : Mengecilkan ukuran gambar (downsampling) dan mengekstrak fitur penting.
- Decoder : Mengembalikan representasi kecil tadi menjadi gambar utuh kembali (upsampling).

1. Input Layer
  encoder_input = keras.Input(shape=(SIZE, SIZE, 3))

2. Encoder (Downsampling Layers)
  def downsample(filters, size, apply_batch_normalization=True):
- Sequence encoder
  x = downsample(16, 4, False)(encoder_input)
  x = downsample(32, 4)(x)
  x = downsample(64, 4, False)(x)
  x = downsample(128, 4)(x)
  x = downsample(256, 4)(x)
  encoder_output = downsample(512, 4)(x)

3. Decoder (Unsampling Layers)
  def upsample(filters, size, apply_dropout=False):
- Sequence decoder
  decoder_input = upsample(512, 4, True)(encoder_output)
  x = upsample(256, 4)(decoder_input)
  x = upsample(128, 4, True)(x)
  x = upsample(64, 4)(x)
  x = upsample(32, 4)(x)
  x = upsample(16, 4)(x)
  x = Conv2DTranspose(8, (2, 2), strides=(1, 1), padding='valid')(x)
  decoder_output = Conv2DTranspose(3, (2, 2), strides=(1, 1), padding='valid')(x)

4. Model Output
   return tf.keras.Model(encoder_input, decoder_output)
   
5. Loss & Optimizer
   model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

**#Loss **
Loss menunjukkan seberapa jauh prediksi model dari target sebenarnya. Dalam kasus autoencoder, loss function yang digunakan adalah Mean Absolute Error (MAE), yang mengukur rata-rata selisih absolut antara piksel gambar hasil rekonstruksi dan gambar target (sketsa).

Loss pada model autoencoder menunjukkan nilai **0.1700** menunjukkan bahwa rata-rata kesalahan piksel per channel cukup kecil, yang berarti model mulai bisa merekonstruksi gambar mendekati target, walaupun belum sempurna.

![image](https://github.com/user-attachments/assets/be72cb24-e9e3-417a-99d6-948d433621f4)






