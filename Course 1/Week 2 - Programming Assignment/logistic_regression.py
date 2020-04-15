import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

'exec(%matplotlib inline)'

# ------------------------- STEP 1: LOADING DATA ------------------------- #

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# Data diatas merupakan data original, mengandung data gambar kucing dengan konfigurasi
# matriks 4 dimensi: (m,num_px,num_px,3), dimana:
#     - m         = jumlah data
#     - num_px    = jumlah pixel gambar => gambar yg digunakan square
#     - 3         = tiga merepresentasikan nilai RGB

# x merupakan input
# y merupakan output

# Data diatas diberi nama orig karena akan matriksnya akan di otak atik

# ------------------------- STEP 2: INISIALISASI DATA ------------------------- #

# Menentukan jumlah data training. diambil dari dimensi [0] matriks original
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]

# Menentukan resolusi gambar
num_px = train_set_x_orig.shape[1]

# Membuat matriks input data dengan menggabungkan seluruh fitur menjadi satu dimensi
# Sehingga matriksnya menjadi (num_px*num_px*3,m) => matriks (rgb values, jumlah data)
# Bentuk matriks:
# | x1_val x2_val x3_val x4_val ... m_val |
# | x1_val x2_val x3_val x4_val ... m_val |
# | x1_val x2_val x3_val x4_val ... m_val |
# | x1_val x2_val x3_val x4_val ... m_val |
# | x1_val x2_val x3_val x4_val ... m_val |
# dst

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten =  test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Cek data dengan print data to console:
# print ("Number of training examples: m_train = " + str(m_train))
# print ("Number of testing examples: m_test = " + str(m_test))
# print ("")
# print ("Height/Width of each image: num_px = " + str(num_px))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("")
# print ("train_set_x shape: " + str(train_set_x_orig.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("")
# print ("test_set_x shape: " + str(test_set_x_orig.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))
# print ("")
# print ("train_set_x flatten: " + str(train_set_x_flatten.shape))
# print ("test_set_x flatten: " + str(test_set_x_flatten.shape))
# print ("")

# ------------------------- STEP 3: PEMBUATAN ALGORITMA ------------------------- #

# Ini merupakan step utama, dimana algoritma Logistic Regression dibuat
# Sebelum membuat fungsi-fungsi untuk digunakan pada Logistic Regression,
# mari review step-step utama algoritma Logistic Regression:
#   1. Menghitung nilai Z
#   2. Melakukan aktivasi Z dengan sigmoid => agar datanya standar dari -1 sampai 1
#   3. Mendapatkan nilai probabilitas
#   4. Memperbaiki nilai probabilitas dengan Gradient Descent (Backward Propagation)

# Step diatas merupakan step utama Logistic Regression. Untuk melakukan Logistic Regression
# dalam bentuk pemrograman, dibutuhkan step-step mendetail. Berikut step-stepnya:
#   0. Menyiapkan fungsi utk membantu perhitungan: 
#      Membuat fungsi untuk men-sigmoid-kan suatu nilai.
#   1. Inisialisasi matriks w & b. 
#   2. Melakukan forward propagation
#       2.1. Menghitung nilai Z
#       2.2. Mensigmoidkan Z
#       2.3. Menghitung nilai kesalahan (Loss Function)
#   3. Melakukan backward propagation
#       3.1. Menghitung gradient w
#       3.2. Menghitung gradient b
#   4. Melakukan optimisasi prediksi
#      Dari nilai gradient yang di dapat, kita dapat merubah nilai w & b agar mendekati
#      gradient = 0. Dimana apabila gradient = 0 maka error = 0.

# ------------------------- STEP 4: MELAKUKAN UJI COBA w & b ------------------------- #
