import numpy as np
from numpy import linalg as LA
from tensorflow.keras.datasets import cifar10
from collections import Counter
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

class KNN:
    def __init__(self):
        self.X= None
        self.Y= None

    def train(self,X,Y,nvec):
        self.im_shape= X.shape[1:]
        self.X= np.int16(np.reshape(X, (X.shape[0],np.prod(self.im_shape))))
        self.Y=Y
        self.n_vec=nvec

    def KNN_1dato(self, x):
        xaux=np.int16(np.reshape(x,(1,np.prod(self.im_shape))))
        distancias = [LA.norm((xaux[0]-self.X[i]),ord=2) for i in range(len(self.X))]
        ind = np.argpartition(distancias,self.n_vec)[:self.n_vec]
        k_clases = np.hstack([self.Y[ind[i]] for i in range(len(ind))])
        return Counter(k_clases).most_common()[0][0]

    def KNN(self, datos_test):
        return ([self.KNN_1dato(datos_test[i]) for i in range(len(datos_test))])

    def accuracy(self,X,Y):
        Yp=self.KNN(X)
        aux=0
        for i in range(len(Yp)):
            if Yp[i]==Y[i]: aux+=1
        return aux/len(Yp)

def Plot_Resultados(vecinos,datosprueba,label,epocas):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    with plt.style.context('seaborn-darkgrid'):
        plt.grid(True)
        k_vecinos = vecinos
        model = KNN()
        model.train(x_train, y_train, k_vecinos)
        acc = model.accuracy(x_test[: datosprueba], y_test[: datosprueba])
        acc_vec=acc*np.ones(len(epocas))
        plt.plot(epocas, acc_vec, label=label)
        plt.ylabel(r'Accuracy test')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        plt.tight_layout()

'''
#Copio datos de cifar10/mnist

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
k_vecinos=3
model=KNN()
model.train(x_train,y_train,k_vecinos)
acc=model.accuracy(x_test[: 200],y_test[: 200])
print(acc)
'''