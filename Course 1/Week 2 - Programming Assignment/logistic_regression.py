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
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Standarisasi data agar bernilai dari 0-1.
# Karena data merupakan RGB values (0-255) maka seluruh data dibagi 255
train_set_x = train_set_x_flatten/255.0
test_set_x = test_set_x_flatten/255.0

# Cek data dengan print data to console:
print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("")
print("train_set_x flatten: " + str(train_set_x_flatten.shape))
print("test_set_x flatten: " + str(test_set_x_flatten.shape))
print("train_set_x final: " + str(train_set_x.shape))
print("test_set_x final: " + str(test_set_x.shape))
print("")

# ------------------------- STEP 3: PEMBUATAN ALGORITMA ------------------------- #

# Pada step ini dilakukan pembuatan fungsi-fungsi pendukung dalam pembuatan model 
# Logistic Regression. Mari review step-step utama algoritma Logistic Regression:
#   1. Menghitung nilai Z --> Z merupakan linear function yg ada di setiap neuron
#   2. Melakukan aktivasi Z dengan sigmoid => agar datanya standar dari -1 sampai 1
#   3. Meghitung loss function => mendapatkan nilai probabilitas
#   4. Memperbaiki nilai probabilitas dengan Gradient Descent (Backward Propagation)
#   5. Update nilai w & b
#   6. Melakukan testing

# Step diatas merupakan step utama Logistic Regression. Untuk melakukan Logistic Regression
# dalam bentuk pemrograman, dibutuhkan step-step mendetail. Berikut step-stepnya:
#   0. Menyiapkan fungsi utk membantu perhitungan:
#      => Membuat fungsi untuk men-sigmoid-kan suatu nilai.
#   1. Inisialisasi matriks w & b.
#   2. Melakukan forward propagation
#       2.1. Menghitung nilai Z
#       2.2. Mensigmoidkan Z
#       2.3. Menghitung nilai kesalahan keseluruhan (Cost Function)
#   3. Melakukan backward propagation
#       3.1. Menghitung gradient w
#       3.2. Menghitung gradient b
#   4. Melakukan optimisasi nilai b & w
#      Dari nilai gradient yang di dapat, kita dapat merubah nilai w & b agar mendekati
#      gradient = 0. Dimana apabila gradient = 0 maka error = 0.
#   5. Mengkonversikan hasil perhitungan model menjadi bentuk akhir data untuk dibandingkan
#      dengan nilai output asli

# ----------- 0. FUNGSI SIGMOID ----------- #
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1/(1+np.exp(-z))

    return s


# ----------- 1. INISIALISASI NILAI w & b ----------- #
def parameter_initialization(dim):
    """
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    w = np.zeros((dim, 1))
    b = 0

    # Cek bentuk matriks w & b => harus matriks vertical
    assert(w.shape == (dim, 1))

    # Cek nilai b. harus float atau int
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b


# ----------- 2&3. FORWARD & BACKWARD PROPAGATION ----------- #
def propagate(w, b, X, Y):
    """
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """

    # Menghitung jumlah data yang di proses
    m = X.shape[1]


    # ----- FORWARD PROPAGATION ----- #
    # Menghitung Z
    z = np.dot(w.T, X) + b

    # Mensigmoidkan Z
    A = sigmoid(z)

    # Menghitung cost function
    cost = -(1/m) * np.sum((Y*np.log(A)) + ((1-Y) * np.log(1-A)))
    
    
    # ----- BACKWARD PROPAGATION ----- #
    # Menghitung turunan J terhadap w (dJ/dw)
    dw = (1/m) * np.dot(X, (A-Y).T)

    # Menghitung turunan J terhadap b (dJ/db)
    db = (1/m) * np.sum(A-Y)

    # Cek bentuk dw, harus sama dengan w
    assert(dw.shape == w.shape)
    assert(db.dtype == float)

    # Cek nilai cost
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


# ----------- 4. OPTIMASI NILAI w & b ----------- #
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    Fungsi ini melakukan optimisasi nilai w & b dengan metode gradient descent
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """

    costs = []

    # Melakukan optimisasi sebanyak X kali
    for i in range(num_iterations):

        # Memperoleh nilai gradient w & b
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        # Update nilai w & b berdasarkan gradientnya
        w = w-(learning_rate*dw)
        b = b-(learning_rate*db)

        # Simpan nilai cost (Error) dan print ke console tiap 100 iterasi
        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    
    # Menyimpan nilai w & b baru yang sudah di optimisasi
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


# ----------- 5. KONVERSI MENJADI BENTUK PREDIKSI FINAL ----------- #
def predict(w, b, X):
    '''
    Fungsi ini berfungsi untuk melakukan uji coba prediksi data menggunakan
    model yang telah dibuat

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    # Jumlah data yang ingin di uji coba
    m = X.shape[1]

    # Membuat matriks hasil prediksi 
    Y_prediction = np.zeros((1,m))

    # Melakukan prediksi data
    z = np.dot(w.T, X) + b
    A = sigmoid(z)

    # Mengkonversi nilai hasil perhitungan menjadi hasil prediksi final
    for i in range (m):
        if(A[0,i] > 0.5):
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    
    
    # Cek bentuk matriks prediksi
    assert(Y_prediction.shape == (1, m))

    return Y_prediction




# ------------------------- STEP 4: MEMBUAT MODEL LOGISTIC REGRESSION BERDASARKAN FUNGSI YANG TELAH DIBUAT ------------------------- #
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Membuat model logistic regression dengan fungsi yang telah di buat pada step 3

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """

    # 1. INISIALISASI DATA DENGAN 0 #
    w,b = parameter_initialization(X_train.shape[0])

    # 2. MELAKUKAN PROSES PERHITUNGAN LOGISTIC REGRESSION #
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # 3. MENYIMPAN PARAMETER w & b #
    w = parameters["w"]
    b = parameters["b"]

    # 4. MELAKUKAN PREDIKSI DATA BERDASARKAN PARAMETER HASIL PERHITUNGAN LOGISTIC REGRESSION #
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # 5. PRINT HASIL PREDIKSI DATA
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


# ------------------------- STEP 5: MELAKUKAN PROSES LOGISTIC REGRESSION BERDASARKAN MODEL YANG TELAH DIBUAT ------------------------- #
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 200000, learning_rate = 0.05, print_cost = True)
