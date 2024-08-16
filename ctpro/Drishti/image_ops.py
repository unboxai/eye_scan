import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
#tf.compat.v1.disable_v2_behavior()
import keras
from tensorflow.keras.layers import Dropout, Dense, Flatten,Conv2D,Activation, Reshape, Permute, Layer, BatchNormalization, ZeroPadding2D, MaxPooling2D, InputLayer
from tensorflow.keras.layers import Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Dense, Flatten,Conv2D,Activation
from tensorflow.keras.models import Model
from  tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import keras
import os
from ctpro.settings import BASE_DIR

# class SelfAttention(Layer):
#     def __init__(self, units=64, input_dim=1024, **kwargs):
#         super(SelfAttention, self).__init__(**kwargs)
#         self.units = units
#         self.input_dim = input_dim
        
#     def build(self, input_shape):
#         b, h, w, c = input_shape
#         self.c_star = max(c // 8, 1)
#         self.wf = self.add_weight(name="f",shape=(self.c_star, c),initializer="random_normal",trainable=True)
#         self.wg = self.add_weight(name="g",shape=(self.c_star, c),initializer="random_normal",trainable=True)
#         self.wh = self.add_weight(name="h",shape=(self.c_star, c),initializer="random_normal",trainable=True)
#         self.wv = self.add_weight(name="v",shape=(self.c_star, c),initializer="random_normal",trainable=True)
#         self.gamma = tf.Variable(name="gamma",initial_value=0.0, dtype=tf.float32, trainable=True)
#         self.flatten_images = Reshape((h*w, c), input_shape=(h, w, c))
#         self.unflatten_images = Reshape((h, w, c), input_shape=(h*w, c))
#         self.permute = Permute((2, 1), input_shape=(h*w, c))
        

    
#     def compute_mask(self, inputs, mask=None):
#         # because the embedding layer before it, supports masking, so this one has to support masking too
#         return None
    
#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def call_prev_(self, inputs, mask=None):
#         x = Lambda(lambda l : self.flatten_images(l))(inputs)
#         x = Lambda(lambda l: self.permute(l))(x)
#         f = Lambda(lambda l: tf.matmul(self.wf, l) )(x)
#         g = Lambda(lambda l: tf.matmul(self.wg, l) )(x)
#         h = Lambda(lambda l: tf.matmul(self.wh, l) )(x) # shape of h = (c_star, h*w) 
#         beta = Lambda(lambda l: tf.matmul(tf.transpose(l[1],perm=[0,2,1]), l[0])) ([g, f])
#         beta = Lambda(lambda l: tf.nn.softmax(l, axis=1))(beta)
#         d = Lambda(lambda l: tf.matmul(tf.transpose(l[0],perm=[0,2,1]), tf.transpose(l[1],perm=[0,2,1])))([beta, h]) # shape will (h*w , c_star)
#         o = Lambda(lambda l: tf.matmul(l, self.wv))(d)
#         o = Lambda(lambda l: tf.multiply(l[1], self.gamma) + tf.transpose(l[0],perm=[0,2,1]))([x, o])
#         o = Lambda(lambda l: self.unflatten_images(l))(o)
#         return o

#     def call(self, inputs, mask=None):
#         x = self.flatten_images(inputs)
#         x = self.permute(x)
#         f = tf.matmul(self.wf, x)
#         g = tf.matmul(self.wg, x)
#         h = tf.matmul(self.wh, x) # shape of h = (c_star, h*w) 
#         beta = tf.matmul(tf.transpose(f,perm=[0,2,1]), g)
#         beta =tf.nn.softmax(beta, axis=1)
#         d = tf.matmul(tf.transpose(beta,perm=[0,2,1]), tf.transpose(h,perm=[0,2,1])) # shape will (h*w , c_star)
#         o = tf.matmul(d, self.wv)
#         o = tf.multiply(o, self.gamma) + tf.transpose(x,perm=[0,2,1])
#         o = self.unflatten_images(o)
#         return o
    

class SelfAttention(Layer):
    def __init__(self, units=64, input_dim=1024, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        # self.gamma = 0.0 # constant float
        
    def build(self, input_shape):
        b, h, w, c = input_shape
        self.c_star = max(c // 8, 1)
        self.wf = self.add_weight(name="f",shape=(self.c_star, c),initializer="random_normal",trainable=True)
        self.wg = self.add_weight(name="g",shape=(self.c_star, c),initializer="random_normal",trainable=True)
        self.wh = self.add_weight(name="h",shape=(self.c_star, c),initializer="random_normal",trainable=True)
        self.wv = self.add_weight(name="v",shape=(self.c_star, c),initializer="random_normal",trainable=True)
        self.gamma = self.add_weight(name="gamma",shape=(1,),initializer="random_normal",trainable=True) # need to register this for new version of tensorflow
        self.flatten_images = Reshape((h*w, c), input_shape=(h, w, c))
        self.unflatten_images = Reshape((h, w, c), input_shape=(h*w, c))
        self.permute = Permute((2, 1), input_shape=(h*w, c))
        

    
    def compute_mask(self, inputs, mask=None):
        # because the embedding layer before it, supports masking, so this one has to support masking too
        return None
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def call_prev_(self, inputs, mask=None):
        x = Lambda(lambda l : self.flatten_images(l))(inputs)
        x = Lambda(lambda l: self.permute(l))(x)
        f = Lambda(lambda l: tf.matmul(self.wf, l) )(x)
        g = Lambda(lambda l: tf.matmul(self.wg, l) )(x)
        h = Lambda(lambda l: tf.matmul(self.wh, l) )(x) # shape of h = (c_star, h*w) 
        beta = Lambda(lambda l: tf.matmul(tf.transpose(l[1],perm=[0,2,1]), l[0])) ([g, f])
        beta = Lambda(lambda l: tf.nn.softmax(l, axis=1))(beta)
        d = Lambda(lambda l: tf.matmul(tf.transpose(l[0],perm=[0,2,1]), tf.transpose(l[1],perm=[0,2,1])))([beta, h]) # shape will (h*w , c_star)
        o = Lambda(lambda l: tf.matmul(l, self.wv))(d)
        o = Lambda(lambda l: tf.multiply(l[1], self.gamma) + tf.transpose(l[0],perm=[0,2,1]))([x, o])
        o = Lambda(lambda l: self.unflatten_images(l))(o)
        return o

    def call(self, inputs, mask=None):
        x = self.flatten_images(inputs)
        x = self.permute(x)
        f = tf.matmul(self.wf, x)
        g = tf.matmul(self.wg, x)
        h = tf.matmul(self.wh, x) # shape of h = (c_star, h*w) 
        beta = tf.matmul(tf.transpose(f,perm=[0,2,1]), g)
        beta =tf.nn.softmax(beta, axis=1)
        d = tf.matmul(tf.transpose(beta,perm=[0,2,1]), tf.transpose(h,perm=[0,2,1])) # shape will (h*w , c_star)
        o = tf.matmul(d, self.wv)
        o = tf.multiply(o, self.gamma) + tf.transpose(x,perm=[0,2,1])
        o = self.unflatten_images(o)
        return o
    


def save_weights_manually(model):
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, "save_weights") or True:
            weight_and_bias = layer.get_weights()
            np.save(f"Drishti/layers_weights/{idx}_layer.npy", weight_and_bias)


def load_weights_manually(model):
    for idx, layer in enumerate(model.layers):
            weight_and_bias = np.load(f"{BASE_DIR}/Drishti/layers_weights/{idx}_layer.npy",  allow_pickle=True)
            print(f"Loading weights for layer # {idx}...\n")
            if isinstance(layer, SelfAttention):
                #lw = layer.get_weights()
                # for k in lw:
                #     print("unit: \n", k, "\n\n")
                # for k in weight_and_bias:
                #     print("unit2: \n", k, "\n\n")
                weight_and_bias[-1] = np.array([weight_and_bias[-1], ])
            layer.set_weights(weight_and_bias)
    return model


# save_weights_manually(model.model)

class DLModel:

    def __init__(self):
        mobile = MobileNetV2()
        # Modify the model
        # Choose the 100th layer from the last
        x = mobile.layers[-50].output
        x = SelfAttention()(x)
        x1 = Conv2D(256, (3,3), padding="same")(x)
        x = Activation("relu")(x1)
        x = BatchNormalization()(x)
        x = SelfAttention()(x)
        x = Conv2D(256, (3,3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = SelfAttention()(x)
        x = Conv2D(256, (3,3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = SelfAttention()(x)
        x = Lambda(lambda l: tf.add(l[0], l[1]))([x1, x])
        x = MaxPooling2D((2,2))(x)
        """
        x = Conv2D(128, (3,3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x2 = SelfAttention()(x)
        x = Conv2D(128, (3,3), padding="same")(x2)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = SelfAttention()(x)
        x = Conv2D(128, (3,3), padding="same")(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)
        x = SelfAttention()(x)
        x = Lambda(lambda l: tf.add(l[0], l[1]))([x2, x])
        x = Conv2D(64, (3,3),padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x3 = SelfAttention()(x)
        x = Conv2D(64, (3,3),padding="same")(x3)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = SelfAttention()(x)
        x = Conv2D(64, (3,3),padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = SelfAttention()(x)
        x = Lambda(lambda l: tf.add(l[0], l[1]))([x3, x])
        x = Conv2D(32, (3,3),padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x4 = SelfAttention()(x)
        x = Conv2D(32, (3,3),padding="same")(x4)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = SelfAttention()(x)
        x = Conv2D(32, (3,3),padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = SelfAttention()(x)
        x = Lambda(lambda l: tf.add(l[0], l[1]))([x4, x])
        x = MaxPooling2D((2,2))(x)
        """
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x2 = Dense(128, activation="relu")(x)
        x2 = BatchNormalization()(x2)
        x2 = Dense(1, activation="sigmoid")(x2)
        self.model = Model(inputs=mobile.inputs, outputs=x2)
        self.model.compile(optimizer=Adam(0.00005), loss="binary_crossentropy",metrics=["accuracy"])
        # self.model.load_weights(r"C:\Users\saura.DESKTOP-LV85RKQ\Documents\Django Project\ctpro\Drishti/weights_cnn.hdf5")
        self.model = load_weights_manually(self.model)
        self.eye_cascade = cv2.CascadeClassifier(r"/home/eye_scan/ctpro/Drishti/eye_haar_cascade.xml")
        
    def find_iris(self, img_path):
        print(f"Loading image from path: {img_path}")  # Add logging statement
        img_ = cv2.imread(img_path)
        if img_ is None:
            print(f"Failed to load image from path: {img_path}")  # Add logging statement
            return None 
        hght = 500
        wdth = int(500*(img_.shape[0]/img_.shape[1]))
        resized_image = cv2.resize(img_, (hght, wdth))
        """if wdth < hght:
            center = int(hght/2)
            resized_image = resized_image[0:wdth,int(center-center/2):int(center+center/2),:]
        elif hght > wdth:
            center = int(wdth/2)
            resized_image = resized_image[int(center-center/2):int(center+center/2),0:hght,:]"""
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray)
        max_area = 0
        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes:
                if ew*eh >= max_area:
                    eroi = gray[ey:ey+eh, ex:ex+ew]
                    eroi2 = resized_image[ey:ey+eh, ex:ex+ew , :]
                    max_area = ew*eh
        else:
              eroi = gray
              eroi2 = resized_image
        img = cv2.medianBlur(eroi, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1.5,10, param1=300, param2=0.8, minRadius=20, maxRadius=150)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            circle = max(circles[0], key=lambda x: x[2])            
            iris = eroi2[max(0, circle[1]-circle[2]) : circle[1]+circle[2], max(0, circle[0]-circle[2]) : circle[0]+circle[2], : ]
            return iris
        else:
            
            if eroi2.shape[0]*eroi2.shape[1] >= 160*160:
              return eroi2
            else:
              return resized_image
    """
    def call(self, image_path):
        iris = self.find_iris(image_path)
        iris = cv2.cvtColor(iris, cv2.COLOR_BGR2RGB)/255.0
        iris_test = cv2.resize(iris, (224, 224))
        iris_test = np.expand_dims(iris_test, 0)
        pred = self.model.predict(iris_test)[0][0]
        print("\n\nCataract Score from Model = ", pred)
        if pred >= 0.21:
            message = 'Normal eye'
        elif (pred < 0.21 and pred >= 0.17):
            message = 'High risk - Get further evaluation'
        else:
            message = f'Cataract found in eye with {round((1 - pred) * 100, 2)} % confidence'
        return message, round((1 - pred) * 100, 2)
    """

    def call(self, image_path, age, symptoms):
        print(symptoms)
        iris = self.find_iris(image_path)
        iris = cv2.cvtColor(iris, cv2.COLOR_BGR2RGB) / 255.0
        iris_test = cv2.resize(iris, (224, 224))
        iris_test = np.expand_dims(iris_test, 0)
        pred = self.model.predict(iris_test)[0][0]
        print("\n\nCataract Score from Model = ", pred)

        age = int(age)

        print(f'Age is {age} and symptoms are {symptoms}')

        if age >= 65:
            score_changed = False
            symptoms = symptoms.lower()  # Convert the entire symptoms string to lowercase
            print(symptoms)
            if "blurred vision" in symptoms:
                pred -= 0.05
                score_changed = True
            if "whitening and cloudiness in vision" in symptoms:
                pred -= 0.08
                score_changed = True
            if "double vision" in symptoms:
                pred -= 0.05
                score_changed = True
            if "multiple images" in symptoms:
                pred -= 0.08
                score_changed = True
            if "halos or light scattering" in symptoms:
                pred -= 0.05
                score_changed = True
            if "pain in eyes" in symptoms:
                pred -= 0.05
                score_changed = True
            if score_changed:
                print('Since age is above 65 and symptoms are present, a slight decrease in probability is done.')
            else:
                print('Although age is above 65, no specified symptoms are present, so no change in probability.')

        if age <= 55:
            pred += 0.05

        if age <=45:
            pred += 0.1

        if pred < 0:
            pred = 0.01
        if pred > 1:
            pred = 0.99

        print(pred)

        print(f'Final prediction score is {pred}')

        if pred >= 0.23:
            message = 'Normal eye'
        elif 0.17 <= pred < 0.23:
            message = 'High risk - Get further evaluation'
        else:
            message = f'Cataract found in eye with {round((1 - pred) * 100, 2)}% confidence'

        return message, round((1 - pred) * 100, 2)




