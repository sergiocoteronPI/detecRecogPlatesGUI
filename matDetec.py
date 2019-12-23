
import tensorflow as tf

import numpy as np
import cv2
import os

from copy import deepcopy

try:
    font = cv2.FONT_HERSHEY_SIMPLEX
except:
    print("Error: No se ha podido ejecutar - cv2.FONT_HERSHEY_SIMPLEX")

class BoundBox:
    def __init__(self, classes):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.label = ''
        self.probs = float()

class claseControladorDetectorDeMatriculas:

    def __init__(self, threshold, batch_size, dim_fil, dim_col, H, W, B, learning_ratio, nms, ver_probs, rpe, rpi, h5):
        
        self.threshold = threshold
        self.batch_size = batch_size

        self.dim_fil = dim_fil
        self.dim_col = dim_col

        self.labels = ['matricula']
        
        self.anchors = [1,1, 1,1, 1,1, 1,1, 1,1, 1,1][0:2*B]

        self.H = H
        self.W = W
        self.C = len(self.labels)
        self.B = B
        self.HW = H*W

        self.colors = np.random.randint(0,255 ,(self.C,3)).tolist()
        self.colors[0] = [255,0,255]
        self.learning_ratio = learning_ratio

        self.nms = nms
        self.ver_probs = ver_probs

        self.clases_visibles = [self.labels.index(v) for v in self.labels]

        self.rpe = rpe
        self.rpi = rpi

        self.h5 = h5

clasMatDetec = claseControladorDetectorDeMatriculas(threshold = 0.5,
                                                    batch_size = 30,
                                                    dim_fil = 480, dim_col = 480,
                                                    H = 13, W = 13, B = 3,
                                                    learning_ratio = 1e-3,
                                                    nms = True,
                                                    ver_probs = True,
                                                    rpe = '/home/sergio/Escritorio/Deep learning/Modelos de visión/Reconocimiento de matrículas/dataset/imageLabel/',
                                                    rpi = '/home/sergio/Escritorio/Deep learning/Modelos de visión/Reconocimiento de matrículas/dataset/imageTrain/',
                                                    h5 = 'mark1_matdetec.h5')

class matDetec():

    def __init__(self):

        if os.path.exists(clasMatDetec.h5):

            print('')
            print('Cargando modelo')
            print('')
            self.model = tf.keras.models.load_model(clasMatDetec.h5, custom_objects={'loss_function': self.loss_function})
            
        else:

            self.model, _ = self.neuralNetwork()
            self.model.compile(loss=self.loss_function,optimizer=tf.keras.optimizers.Adam(lr = 0.001))

        print('')
        print(self.model.summary())
        print('')

    def matDetecFunction(self, imagen):

        _imagen = deepcopy(imagen)

        try:
            origShapeY, origShapeX, _ = _imagen.shape
            
            multY, multX = origShapeY, origShapeX
            if orig_y < clasMatDetec.dim_fil:
                multY = clasMatDetec.dim_fil
            if orig_x < clasMatDetec.dim_col:
                multX = clasMatDetec.dim_col
        except:
            return None, []

        frameAdaptado = self.retocar(_imagen)
        frameNormalizado = (frameAdaptado/255)*2 - 1
                    
        neuralNetworkOut = self.model.predict(x=np.array([frameNormalizado]))
        box, imgOut = self.postprocess(neuralNetworkOut, imagen, multY, multX)

        return imgOut, box


    # Pre procesamiento de la imagen para darsela a la red neuronal #
    # ============================================================= #
    def retocar(self, img):
    
        zeros = np.zeros([clasMatDetec.dim_fil,clasMatDetec.dim_col,3])
        im_sha_1, im_sha_2, _ = img.shape
        if im_sha_1 >= clasMatDetec.dim_fil:
            if im_sha_2 >= clasMatDetec.dim_col:
                try:
                    zeros = cv2.resize(img,(clasMatDetec.dim_col,clasMatDetec.dim_fil))
                except:
                    return None
            else:
                try:
                    zeros[:,0:im_sha_2,:] = cv2.resize(img,(im_sha_2,clasMatDetec.dim_fil))
                except:
                    return None
        elif im_sha_2 >= clasMatDetec.dim_col:
            try:
                zeros[0:im_sha_1,:,:] = cv2.resize(img,(clasMatDetec.dim_col,im_sha_1))
            except:
                return None
        else:
            zeros[0:im_sha_1, 0:im_sha_2,:] = img
        return zeros
    # ============================================================= #

    # Funcion pérdida y función para calcular la intersección sobre la unión #
    # ====================================================================== #

    def expit_tensor(self, x):
        return 1. / (1. + tf.exp(-tf.clip_by_value(x,-10,10)))

    def calc_iou(self, boxes1, boxes2):

        boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                        boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                        boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                        boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

        boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                        boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                        boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                        boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
        boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

        lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)
            
    def loss_function(self, yTrue, yPred):
        
        sprob = 1
        sconf = 1
        snoob = 0.25
        scoor = 5
        
        H, W = clasMatDetec.H, clasMatDetec.W
        B, C = clasMatDetec.B, clasMatDetec.C
        
        anchors = clasMatDetec.anchors

        _coord = tf.reshape(yTrue[:,:,:,:B*4], [-1, H*W, B, 4])
        _confs = tf.reshape(yTrue[:,:,:,B*4:B*5], [-1, H*W, B])
        _probs = tf.reshape(yTrue[:,:,:,B*5:], [-1, H*W, B, C])

        _uno_obj = tf.reshape(tf.minimum(tf.reduce_sum(_confs, [2]), 1.0),[-1, H*W])

        net_out_coords = tf.reshape(yPred[:,:,:,:B*4], [-1, H*W, B, 4])
        net_out_confs = tf.reshape(yPred[:,:,:,B*4:B*5], [-1, H, W, B])
        net_out_probs = tf.reshape(yPred[:,:,:,B*5:], [-1, H, W, B, C])
                                                               
        adjusted_coords_xy = self.expit_tensor(net_out_coords[:,:,:,0:2])
        adjusted_coords_wh = tf.sqrt(tf.exp(tf.clip_by_value(net_out_coords[:,:,:,2:4],-15,8))* np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
        adjusted_coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)
        
        x_yolo = tf.reshape(adjusted_coords_xy[:,:,:,0],[-1,H*W,B])
        y_yolo = tf.reshape(adjusted_coords_xy[:,:,:,1],[-1,H*W,B])
        w_yolo = tf.reshape(adjusted_coords_wh[:,:,:,0],[-1,H*W,B])
        h_yolo = tf.reshape(adjusted_coords_wh[:,:,:,1],[-1,H*W,B])
        
        adjusted_c = self.expit_tensor(net_out_confs)
        adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B])
        
        adjusted_prob = self.expit_tensor(net_out_probs)
        adjusted_prob = tf.reshape(adjusted_prob,[-1, H*W, B, C])
        
        iou = self.calc_iou(tf.reshape(_coord, [-1, H, W, B, 4]), tf.reshape(adjusted_coords,[-1, H, W, B, 4]))
        best_box = tf.reduce_max(iou, 3, keepdims=True)
        best_box = tf.to_float(best_box)
        confs = tf.reshape(tf.cast((iou >= best_box), tf.float32),[-1,H*W,B]) * _confs

        coord_loss_xy = scoor*tf.reduce_mean(tf.reduce_sum(_confs*(tf.reshape(tf.square(x_yolo - _coord[:,:,:,0]) + tf.square(y_yolo - _coord[:,:,:,1]),[-1,H*W,B])),[1,2]))# + \
        coord_loss_wh = scoor*tf.reduce_mean(tf.reduce_sum(_confs*(tf.reshape(tf.square(w_yolo - _coord[:,:,:,2]) + tf.square(h_yolo - _coord[:,:,:,3]),[-1,H*W,B])),[1,2]))# + \
        
        conf_loss = sconf*tf.reduce_mean(tf.reduce_sum(_confs*tf.square(adjusted_c - confs),[1,2])) + \
                    snoob*tf.reduce_mean(tf.reduce_sum((1.0 - _confs)*tf.square(adjusted_c - confs),[1,2]))
        
        class_loss = sprob*tf.reduce_mean(tf.reduce_sum(_uno_obj*tf.reduce_sum(tf.square(adjusted_prob - _probs),[2,3]),1))
     
        loss = coord_loss_xy + coord_loss_wh + class_loss + conf_loss

        return loss

    # ====================================================================== #

    # Red neuronal para la deteción de matrículas en imágenes. También están aquí las funciones necesarias para que funcione #
    # ====================================================================================================================== #

    def conv2d(self, inputs, f = 32, k = (3,3), s = 1, activation=None, padding = 'valid'):

        return tf.keras.layers.Conv2D(filters = f, kernel_size = k ,strides=(s, s),
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


    def neuralNetwork(self):

        x = tf.keras.Input(shape=(clasMatDetec.dim_fil,clasMatDetec.dim_col,3), name='input_layer')

        h_c1 = self.conv2d(inputs = x, f = 8, k = (3,3), s = 2, padding='same')
        pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=None,padding='valid')(x)
        h_c1 = tf.keras.layers.concatenate([pool1, h_c1])

        h_c1 = self.conv2d(inputs = h_c1, f = 16, k = (3,3), s = 2, padding='same')
        pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=None,padding='valid')(pool1)
        h_c1 = self.leaky_relu(tf.keras.layers.concatenate([pool2, h_c1]))

        h_c1 = self.batch_norm(self.conv2d(inputs = h_c1, f = 32, k = (3,3), s = 2))

        h_c1 = self.dense_layer(h_c1, 16, 32)
        h_c1 = self.leaky_relu(self.batch_norm(self.conv2d(inputs = h_c1, f = 512, k = (3,3), s = 1)))

        h_c1 = self.dense_layer(h_c1, 32, 64)

        h_c1 = self.conv2d(inputs = h_c1, f = clasMatDetec.B*(4+1+clasMatDetec.C), k = (1,1), s = 1)

        model = tf.keras.Model(inputs=x, outputs=h_c1)

        return model, h_c1

    # ====================================================================================================================== # 


    # Post procesamiento. Tomamos lo devuelto por la red neuronal y lo convertimos en una imagen de salida y un array con los datos encontrados #
    # ========================================================================================================================================= #

    def overlap_c(self, x1, w1 , x2 , w2):
        l1 = x1 - w1 /2.
        l2 = x2 - w2 /2.
        left = max(l1,l2)
        r1 = x1 + w1 /2.
        r2 = x2 + w2 /2.
        right = min(r1, r2)
        return right - left

    def box_intersection_c(self, ax, ay, aw, ah, bx, by, bw, bh):
        w = self.overlap_c(ax, aw, bx, bw)
        h = self.overlap_c(ay, ah, by, bh)
        if w < 0 or h < 0: return 0
        area = w * h
        return area

    def box_union_c(self, ax, ay, aw, ah, bx, by, bw, bh):
        i = self.box_intersection_c(ax, ay, aw, ah, bx, by, bw, bh)
        u = aw * ah + bw * bh -i
        return u

    def box_iou_c(self, ax, ay, aw, ah, bx, by, bw, bh):
        return self.box_intersection_c(ax, ay, aw, ah, bx, by, bw, bh) / self.box_union_c(ax, ay, aw, ah, bx, by, bw, bh)

    def expit_c(self, x):
        return 1/(1+np.exp(-np.clip(x,-10,10)))
        
    def NMS(self, final_probs , final_bbox):

        labels, C = clasMatDetec.labels, clasMatDetec.C
        
        boxes = []
        indices = []
    
        pred_length = final_bbox.shape[0]
        class_length = final_probs.shape[1]

        for class_loop in range(class_length):
            for index in range(pred_length):
                if final_probs[index,class_loop] == 0: continue
                
                for index2 in range(index+1,pred_length):
                    if final_probs[index2,class_loop] == 0: continue
                    if index==index2 : continue
                    
                    if self.box_iou_c(final_bbox[index,0],final_bbox[index,1],final_bbox[index,2],final_bbox[index,3],
                                      final_bbox[index2,0],final_bbox[index2,1],final_bbox[index2,2],final_bbox[index2,3]) >= 0.1:
                        if final_probs[index2,class_loop] > final_probs[index, class_loop] :
                            final_probs[index, class_loop] = 0
                            break
                        final_probs[index2,class_loop]=0
                if index not in indices:

                    bb=BoundBox(C)
                    bb.x = final_bbox[index, 0]
                    bb.y = final_bbox[index, 1]
                    bb.w = final_bbox[index, 2]
                    bb.h = final_bbox[index, 3]

                    bb.label = labels[class_loop]
                    bb.probs = final_probs[index,class_loop]

                    boxes.append(bb)
                    
                    indices.append(index)
                    
        return boxes

    def box_constructor(self, net_out_in):

        threshold, anchors = clasMatDetec.threshold, clasMatDetec.anchors

        H, W = clasMatDetec.H, clasMatDetec.W
        B, C = clasMatDetec.B, clasMatDetec.C

        Bbox_pred = net_out_in[:,:,:,:B*4].reshape([H, W, B,4])
        Conf_pred = net_out_in[:,:,:,B*4:B*5].reshape([H, W, B])
        Classes = net_out_in[:,:,:,B*5:].reshape([H, W, B, C])
        
        probs = np.zeros((H, W, B, C), dtype=np.float32)
        _Bbox_pred = np.zeros((H, W, B, 5), dtype=np.float32)
        
        for row in range(H):
            for col in range(W):
                for box_loop in range(B):

                    Classes[row, col, box_loop, :] = self.expit_c(Classes[row, col, box_loop, :])
                    if np.max(Classes[row, col, box_loop, :]) < threshold:
                        continue
                
                    Conf_pred[row, col, box_loop,] = self.expit_c(Conf_pred[row, col, box_loop])
                    if Conf_pred[row, col, box_loop] < threshold:
                        continue
                    
                    Bbox_pred[row, col, box_loop, 0] = (col + self.expit_c(Bbox_pred[row, col, box_loop, 0])) / W
                    Bbox_pred[row, col, box_loop, 1] = (row + self.expit_c(Bbox_pred[row, col, box_loop, 1])) / H
                    Bbox_pred[row, col, box_loop, 2] = np.exp(np.clip(Bbox_pred[row, col, box_loop, 2],-15,8)) * anchors[2 * box_loop + 0] / W
                    Bbox_pred[row, col, box_loop, 3] = np.exp(np.clip(Bbox_pred[row, col, box_loop, 3],-15,8)) * anchors[2 * box_loop + 1] / H

                    for class_loop in range(C):

                        tempc = Classes[row, col, box_loop, class_loop] * Conf_pred[row, col, box_loop]
                        if(tempc > threshold):

                            probs[row, col, box_loop, class_loop] = tempc
                            _Bbox_pred[row, col, box_loop, 0] = Bbox_pred[row, col, box_loop, 0]
                            _Bbox_pred[row, col, box_loop, 1] = Bbox_pred[row, col, box_loop, 1]
                            _Bbox_pred[row, col, box_loop, 2] = Bbox_pred[row, col, box_loop, 2]
                            _Bbox_pred[row, col, box_loop, 3] = Bbox_pred[row, col, box_loop, 3]
                            _Bbox_pred[row, col, box_loop, 4] = Conf_pred[row, col, box_loop]
                            
        return self.NMS(np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(_Bbox_pred).reshape(H*W*B,5))

    def box_constructor_sin_nms(self, net_out_in):

        threshold = clasMatDetec.threshold
        labels = clasMatDetec.labels
        anchors = clasMatDetec.anchors

        H, W = clasMatDetec.H, clasMatDetec.W
        B, C = clasMatDetec.B, clasMatDetec.C
        
        boxes = []

        Bbox_pred = net_out_in[:,:,:,:B*4].reshape([H, W, B,4])
        Conf_pred = net_out_in[:,:,:,B*4:B*5].reshape([H, W, B])
        Classes = net_out_in[:,:,:,B*5:].reshape([H, W, B, C])
        
        _Bbox_pred = np.zeros((H, W, B, 5), dtype=np.float32)
        
        for row in range(H):
            for col in range(W):
                for box_loop in range(B):

                    Classes[row, col, box_loop, :] = self.expit_c(Classes[row, col, box_loop, :])
                    if np.max(Classes[row, col, box_loop, :]) < threshold:
                        continue
                
                    Conf_pred[row, col, box_loop,] = self.expit_c(Conf_pred[row, col, box_loop])
                    if Conf_pred[row, col, box_loop] < threshold:
                        continue
                    
                    Bbox_pred[row, col, box_loop, 0] = (col + self.expit_c(Bbox_pred[row, col, box_loop, 0])) / W
                    Bbox_pred[row, col, box_loop, 1] = (row + self.expit_c(Bbox_pred[row, col, box_loop, 1])) / H
                    Bbox_pred[row, col, box_loop, 2] = np.exp(np.clip(Bbox_pred[row, col, box_loop, 2],-15,8)) * anchors[2 * box_loop + 0] / W
                    Bbox_pred[row, col, box_loop, 3] = np.exp(np.clip(Bbox_pred[row, col, box_loop, 3],-15,8)) * anchors[2 * box_loop + 1] / H

                    for class_loop in range(C):
                        
                        tempc = Classes[row, col, box_loop, class_loop] * Conf_pred[row, col, box_loop]
                        if(tempc > threshold):

                            bb=BoundBox(C)

                            bb.x = Bbox_pred[row, col, box_loop, 0]
                            bb.y = Bbox_pred[row, col, box_loop, 1]
                            bb.w = Bbox_pred[row, col, box_loop, 2]
                            bb.h = Bbox_pred[row, col, box_loop, 3]

                            bb.label = labels[class_loop]
                            bb.probs = tempc

                            boxes.append(bb)

                            
        return boxes

    def findboxes(self, net_out):
        
        boxes = []
        if clasMatDetec.nms:
            boxes = self.box_constructor(net_out)
        else:
            boxes = self.box_constructor_sin_nms(net_out)
        
        return boxes

    def process_box(self, b, h, w):
        max_prob = b.probs
        label = b.label
        if max_prob > clasMatDetec.threshold:
            left  = int ((b.x - b.w/2.) * w)
            right = int ((b.x + b.w/2.) * w)
            top   = int ((b.y - b.h/2.) * h)
            bot   = int ((b.y + b.h/2.) * h)
            if left  < 0    :  left = 0
            if right > w - 1: right = w - 1
            if top   < 0    :   top = 0
            if bot   > h - 1:   bot = h - 1
            mess = '{}'.format(label)
            return (left, right, top, bot, mess, max_prob)
        return None


    def postprocess(self, net_out, im, h, w):

        labels = clasMatDetec.labels
        colors = clasMatDetec.colors

        boxes = self.findboxes(net_out)
        
        imgcv = im.astype('uint8')

        resultsForJSON = []
        for b in boxes:
            
            boxResults = self.process_box(b, h, w)
            if boxResults is None:
                continue
            
            left, right, top, bot, mess, confidence = boxResults
            resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})

            try:
                cv2.rectangle(imgcv,(left, top), (right, bot),colors[labels.index(mess)], 2)
            except:
                print("los cv2 en try-except")
                
            confi = confidence*100

            if clasMatDetec.ver_probs:
                if top - 16 > 0:
                    try:
                        cv2.rectangle(imgcv,(left-1, top - 16), (left + (len(mess)+9)*5*2-1, top),colors[labels.index(mess)], -1)
                        cv2.putText(imgcv,mess + ' -> ' + "%.2f" % confi  + '%' ,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)
                    except:
                        print("los cv2 en try-except")
                else:
                    try:
                        cv2.rectangle(imgcv,(left-1, top), (left + (len(mess)+9)*5*2-1, top+16),colors[labels.index(mess)], -1)
                        cv2.putText(imgcv,mess + ' -> ' + "%.2f" % confi  + '%' ,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)
                    except:
                        print("los cv2 en try-except")
            else:
                if top - 16 > 0:
                    try:
                        cv2.rectangle(imgcv,(left-1, top - 16), (left + len(mess)*5*2-1, top),colors[labels.index(mess)], -1)
                        cv2.putText(imgcv,mess,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)
                    except:
                        print("los cv2 en try-except")
                else:
                    try:
                        cv2.rectangle(imgcv,(left-1, top), (left + len(mess)*5*2-1, top+16),colors[labels.index(mess)], -1)
                        cv2.putText(imgcv,mess,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)
                    except:
                        print("los cv2 en try-except")

        return resultsForJSON, imgcv

    # ========================================================================================================================================= #