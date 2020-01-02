
import tensorflow as tf

import numpy as np
import string
import cv2
import os

from copy import deepcopy

try:
    font = cv2.FONT_HERSHEY_SIMPLEX
except:
    print("Error: No se ha podido ejecutar - cv2.FONT_HERSHEY_SIMPLEX")

class claseControladorReconocimientoDeMatriculas:

    def __init__(self, threshold, batch_size, dim_fil, dim_col, learning_ratio, rpi, h5):
        
        self.threshold = threshold
        self.batch_size = batch_size

        self.dim_fil = dim_fil
        self.dim_col = dim_col

        self.learning_ratio = learning_ratio

        self.rpi = rpi

        self.h5 = h5

        _dict = [v for v in string.ascii_uppercase]
        for i in range(10):
            _dict.append(str(i))
        self.dict = _dict





clasMatOcr = claseControladorReconocimientoDeMatriculas(threshold = 0.5,
                                                          batch_size = 5,
                                                          dim_fil = 32, dim_col = 128,
                                                          learning_ratio = 1e-3,
                                                          rpi = 'dataset/',
                                                          h5 = 'mark1_matocr.h5')





class matRecog():

    # init para la carga de la red neuronal al llamar a la clase #
    # ========================================================== #

    def __init__(self):

        if os.path.exists(clasMatOcr.h5):

            print('')
            print(' ===== Cargando modelo OCR =====')
            print('')
            
            #self.model = tf.keras.models.load_model(clasMatOcr.h5, custom_objects={'loss_function': self.loss_function})

            x, h_out = self.neuralNetwork()
            self.model = tf.keras.Model(inputs=x, outputs=h_out)
            self.model.compile(loss=self.loss_function, optimizer=tf.keras.optimizers.Adam(lr = 0.001))
            
            self.model.load_weights(clasMatOcr.h5)
            
        else:

            x, h_out = self.neuralNetwork()
            self.model = tf.keras.Model(inputs=x, outputs=h_out)
            self.model.compile(loss=self.loss_function, optimizer=tf.keras.optimizers.Adam(lr = 0.001))

        print('')
        print(self.model.summary())
        print('')

    # ========================================================== #

    # Función que dado el recorte de la matrícula analiza y devuelve el string encontrado #
    # =================================================================================== #

    def matOCRFunction(self, imagen):

        _imagen = deepcopy(imagen)

        frameAdaptado = self.retocar(_imagen)
        frameNormalizado = frameAdaptado#/255)*2 - 1
     
        neuralNetworkOut = self.model.predict(x=np.array([frameNormalizado]))
        neuralNetworkOutReorganized = np.transpose(neuralNetworkOut, (1, 0, 2))

        decoded, logProb = self.decode(neuralNetworkOutReorganized, 1*[32])        
        decoded = tf.to_int32(decoded[0])

        decodedTraduce = tf.keras.backend.eval(decoded)[0]

        predictionOCR = self.traducir(decodedTraduce)

        return predictionOCR

    # =================================================================================== #

    # Pre procesamiento de la imagen para darsela a la red neuronal #
    # ============================================================= #
    def retocar(self, img):
    
        zeros = np.zeros([clasMatOcr.dim_fil,clasMatOcr.dim_col])
        im_sha_1, im_sha_2 = img.shape
        if im_sha_1 >= clasMatOcr.dim_fil:
            if im_sha_2 >= clasMatOcr.dim_col:
                try:
                    zeros = cv2.resize(img,(clasMatOcr.dim_col,clasMatOcr.dim_fil))
                except:
                    return None
            else:
                try:
                    zeros[:,0:im_sha_2] = cv2.resize(img,(im_sha_2,clasMatOcr.dim_fil))
                except:
                    return None
        elif im_sha_2 >= clasMatOcr.dim_col:
            try:
                zeros[0:im_sha_1,:] = cv2.resize(img,(clasMatOcr.dim_col,im_sha_1))
            except:
                return None
        else:
            zeros[0:im_sha_1, 0:im_sha_2] = img
        return zeros
    # ============================================================= #

    # Red neuronal para el OCR de matriculas. También están aquí las funciones necesarias para que funcione y la función pérdida y el decode #
    # ====================================================================================================================================== #

    def conv2d(self, inputs, f = 32, k = (3,3), s = 1, activation=None, padding = 'valid'):

        return tf.keras.layers.Conv2D(filters = f, kernel_size = k ,strides=(s, s),
                                    padding=padding,
                                    activation=activation,
                                    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(inputs)

    def conv1d(self, inputs, f = 32, k = 3, s = 1, activation=None, padding = 'valid'):

        return tf.keras.layers.Conv1D(filters = f, kernel_size = k ,strides=s,
                                    padding=padding,
                                    activation=activation,
                                    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(inputs)                                  
        
    def leaky_relu(self, inputs, alpha = 0.2):
        
        return tf.keras.layers.LeakyReLU()(inputs)

    def dropout(self, inputs, keep_prob):

        return tf.keras.layers.Dropout(keep_prob)(inputs)

    def Flatten(self, inputs):
        
        return tf.keras.layers.Flatten()(inputs)

    def Dense(self, inputs, units = 1024, use_bias = True, activation = None):
        
        return tf.keras.layers.Dense(units,activation=activation,use_bias=True,)(inputs)

    def batch_norm(self, inputs):
        
        return tf.keras.layers.BatchNormalization(axis=-1,
                                                momentum=0.99,
                                                epsilon=0.001,
                                                center=True,
                                                scale=True,
                                                beta_initializer='zeros',
                                                gamma_initializer='ones',
                                                moving_mean_initializer='zeros',
                                                moving_variance_initializer='ones')(inputs)

    def dense_layer(self, input_, reduccion, agrandamiento):

        dl_1 = self.conv2d(inputs = input_, f = reduccion, k = (1,1), s = 1)
        dl_1 = self.conv2d(inputs = dl_1, f = agrandamiento, k = (3,3), s = 1, padding = 'same')
        dl_1 = self.leaky_relu(tf.keras.layers.concatenate([input_, dl_1]))

        dl_2 = self.conv2d(inputs = dl_1, f = reduccion, k = (1,1), s = 1)
        dl_2 = self.conv2d(inputs = dl_2, f = agrandamiento, k = (3,3), s = 1, padding = 'same')
        dl_1 = self.leaky_relu(tf.keras.layers.concatenate([dl_1, dl_2]))

        dl_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=None,padding='valid')(dl_1)

        dl_2 = self.conv2d(inputs = dl_1, f = reduccion, k = (1,1), s = 1)
        dl_2 = self.conv2d(inputs = dl_2, f = agrandamiento, k = (3,3), s = 1, padding = 'same')
        dl_1 = self.leaky_relu(tf.keras.layers.concatenate([dl_1, dl_2]))

        dl_2 = self.conv2d(inputs = dl_1, f = reduccion, k = (1,1), s = 1)
        dl_2 = self.conv2d(inputs = dl_2, f = agrandamiento, k = (3,3), s = 1, padding = 'same')
        dl_1 = self.leaky_relu(tf.keras.layers.concatenate([dl_1, dl_2]))

        return dl_1

    def traducir(self, a_traducir):
        frase = ''
        for i in a_traducir:
            if i == 200:
                break
            frase += clasMatOcr.dict[i]
        return frase

    def decode(self, inputs, sequence_length):

        return tf.nn.ctc_greedy_decoder(inputs, sequence_length=sequence_length)#features['seq_lens'])

    def loss_function(self, yTrue, yPred):

        yTrue = tf.cast(yTrue, dtype = tf.int32)
        yTrue = tf.contrib.layers.dense_to_sparse(yTrue,eos_token=200)

        return tf.nn.ctc_loss(labels = yTrue,
                              inputs = yPred,
                              sequence_length = clasMatOcr.batch_size*[32],
                              time_major=False)


    def neuralNetwork(self):

        x = tf.keras.Input(shape=(clasMatOcr.dim_fil,clasMatOcr.dim_col), name='input_layer')
        h_c1 = tf.keras.layers.Permute((2,1))(x)

        h_c1 = self.leaky_relu(self.batch_norm(self.conv1d(inputs = h_c1, f = 64, k = 5, s = 2, padding = 'same')))
        h_c1 = self.leaky_relu(self.batch_norm(self.conv1d(inputs = h_c1, f = 128, k = 3, s = 1, padding = 'same')))
        h_c1 = self.leaky_relu(self.batch_norm(self.conv1d(inputs = h_c1, f = 256, k = 3, s = 2, padding = 'same')))
        h_c1 = self.leaky_relu(self.batch_norm(self.conv1d(inputs = h_c1, f = 512, k = 3, s = 1, padding = 'same')))

        h_c1 = self.leaky_relu(self.batch_norm(self.conv1d(inputs = h_c1, f = 1024, k = 3, s = 1, padding = 'same')))

        h_c1 = tf.keras.layers.Dropout(0.)(h_c1)
        
        h_c1 = tf.keras.layers.Dense(1024)(h_c1)
        h_c1 = tf.keras.layers.Dense(len(clasMatOcr.dict) + 1)(h_c1)

        return x, h_c1

    # ====================================================================================================================== #