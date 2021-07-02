import numpy as np
import models
import activations
import optimizers
import layers
import losses
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import tp1p5 as p5
import tp1p3 as p3


def Plot_Resultados(epocas,loss,ac,ac_test,label):
    with plt.style.context('seaborn-darkgrid'):
        plt.grid(True)

        ax1 = plt.subplot(311)
        plt.plot(epocas,loss, label=label)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        ax2 = plt.subplot(312)
        plt.plot(epocas,ac, label=label)
        plt.ylabel(r'Accuracy')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        plt.tight_layout()
        ax3 = plt.subplot(313)
        plt.plot(epocas, ac_test, label=label)
        plt.ylabel(r'Accuracy test')
        plt.xlabel(r'Epocas transcurridas')

        plt.legend()
        plt.tight_layout()

def Plot_Comparacion(epocas, ac_test_mse, label1):
    with plt.style.context('seaborn-darkgrid'):
        plt.grid(True)
        plt.plot(epocas, ac_test_mse, label=label1)
        plt.ylabel(r'Accuracy test')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        plt.tight_layout()



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train=np.int16(np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:]))))/255
x_test=np.int16(np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:]))))/255
#x_train=x_train-np.mean(x_train,axis=0)
#x_test=x_test-np.mean(x_train,axis=0)
y_train=y_train.reshape(len(y_train))
y_test=y_test.reshape(len(y_test))
y_train=y_train.astype(np.int16)
y_test=y_test.astype(np.int16)
test_data=[x_test, y_test]
'''
x_train=x_train[:10000]
y_train=y_train[:10000]
x_test=x_test[:1000]
y_test=y_test[:1000]
'''
lr=[1e-2,1e-3,1e-4,1e-5,1e-6]
lamd=[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
bs_v=[32,64,128,200]
bs=200
lam=1e-4
epocas=20
opt=optimizers.BGD(5e-5,bs,lam)
model=models.Network(opt)
layer1=layers.Dense(100,activations.Sigmoid(),len(x_train[0]),bs)
layer1_bis=layers.Dense(100,activations.Sigmoid(),100,bs)
model.add(layer1)
model.add(layer1_bis)
layer2=layers.Dense(10,activations.Lineal(),100,bs)
model.add(layer2)
ep,loss,acc,ac_t=model.fit(x_train,y_train,test_data=test_data,bs=bs,epochs=epocas,loss_fun=losses.CCE(),flag='Problema CIFAR',lambd=lam)
Plot_Resultados(ep,loss,acc,ac_t,label='Sigmoide | CCE')
lam=1e-5
opt=optimizers.BGD(5e-5,bs,lam)
model=models.Network(opt)
layer1=layers.Dense(100,activations.Sigmoid(),len(x_train[0]),bs)
layer1_bis=layers.Dense(100,activations.Sigmoid(),100,bs)
model.add(layer1)
model.add(layer1_bis)
layer2=layers.Dense(10,activations.Lineal(),100,bs)
model.add(layer2)


ep2,loss2,acc2,ac_t2=model.fit(x_train,y_train,test_data=test_data,bs=bs,epochs=epocas,loss_fun=losses.MSE(),flag='Problema CIFAR',lambd=lam)
Plot_Resultados(ep2,loss2,acc2,ac_t2,label='Sigmoide | MSE')

bs=200
lam=1e-7

opt=optimizers.BGD(1e-6,bs,lam)
model=models.Network(opt)
layer1=layers.Dense(100,activations.ReLu(),len(x_train[0]),bs)
layer1_bis=layers.Dense(100,activations.ReLu(),100,bs)
model.add(layer1)
model.add(layer1_bis)
layer2=layers.Dense(10,activations.Lineal(),100,bs)
model.add(layer2)

ep3,loss3,acc3,ac_t3=model.fit(x_train,y_train,test_data=test_data,bs=bs,epochs=epocas,loss_fun=losses.CCE(),flag='Problema CIFAR',lambd=lam)
Plot_Resultados(ep3,loss3,acc3,ac_t3,label='ReLU | CCE')
lam=1e-7
opt=optimizers.BGD(1e-6,bs,lam)
model=models.Network(opt)
layer1=layers.Dense(100,activations.ReLu(),len(x_train[0]),bs)
layer1_bis=layers.Dense(100,activations.ReLu(),100,bs)
model.add(layer1)
model.add(layer1_bis)
layer2=layers.Dense(10,activations.Lineal(),100,bs)
model.add(layer2)


ep4,loss4,acc4,ac_t4=model.fit(x_train,y_train,test_data=test_data,bs=bs,epochs=epocas,loss_fun=losses.MSE(),flag='Problema CIFAR',lambd=lam)
Plot_Resultados(ep4,loss4,acc4,ac_t4,label='ReLU | MSE')
plt.show()

Plot_Comparacion(ep,ac_t,'Sigmoide|Sigmoide|Lineal|CCE')
Plot_Comparacion(ep2,ac_t2,'Sigmoide|Sigmoide|Lineal|MSE')
Plot_Comparacion(ep3,ac_t3,'ReLU|ReLU|Lineal|CCE')
Plot_Comparacion(ep4,ac_t4,'ReLU|ReLU|Lineal|MSE')

opt=optimizers.BGD(1e-6,200,1e-7)
model=models.Network(opt)
layer1=layers.Dense(100,activations.Sigmoid(),len(x_train[0]),bs)
model.add(layer1)
layer2=layers.Dense(10,activations.Lineal(),100,bs)
model.add(layer2)
ep5,loss5,acc5,ac_t5=model.fit(x_train,y_train,test_data=test_data,bs=200,epochs=epocas,loss_fun=losses.MSE(),flag='Problema CIFAR',lambd=1e-5)
Plot_Comparacion(ep5,ac_t5,'Punto 3: Sigmoide|Lineal|MSE')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
p5.Plot_Resultados(10, 1, 1e-5, 32, 1e-3, x_train, y_train, x_test, y_test, len(x_test), 'SVM', 'SVM')
p5.Plot_Resultados(10, 0, 1e-3, 32, 1e-2, x_train, y_train, x_test, y_test, len(x_test), 'SM', 'SM')
plt.ylabel(r'Accuracy test')
plt.xlabel(r'Epocas transcurridas')
plt.legend()
plt.tight_layout()
p3.Plot_Resultados(3,100,'KNN con K=3',ep)
#p3.Plot_Resultados(3,len(x_test),'KNN con K=3',ep)
plt.show()
