import numpy as np
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import tp1p5 as p5

def scores(W,x):
    return x.dot(W)

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def grad_sigmoid(x):
    return np.exp(-1*x)/(1+np.exp(-1*x))**2

def MSE(scores,y_true):
    return np.mean(np.sum((scores-y_true)**2,axis=1))

def grad_MSE(scores,y_true):
    return 2*(scores-y_true)

def accuracy_rn(s,y_true):
    y_pred=np.argmax(s, axis=1)
    y_true=np.argmax(y_true, axis=1)
    acc = (y_pred==y_true).mean()
    return acc

def Plot_Resultados(epocas,loss,ac,ac_test,label,modo):
    with plt.style.context('seaborn-darkgrid'):
        plt.grid(True)
        if modo == 'resultados':
            ax1 = plt.subplot(311)
            plt.plot(epocas,loss, 'o-', label=label)
            plt.ylabel(r'Loss')
            plt.xlabel(r'Epocas transcurridas')
            plt.legend()
            ax2 = plt.subplot(312)
            plt.plot(epocas,ac, 'o-', label=label)
            plt.ylabel(r'Accuracy train')
            plt.xlabel(r'Epocas transcurridas')
            plt.legend()
            plt.tight_layout()
            ax3 = plt.subplot(313)
            plt.plot(epocas, ac_test, 'o-', label=label)
            plt.ylabel(r'Accuracy test')
            plt.xlabel(r'Epocas transcurridas')

        if modo=='comparacion':
            plt.plot(epocas, ac_test, 'o-', label=label)
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            p5.Plot_Resultados(10, 1, 1e-5, 32, 1e-3, x_train, y_train, x_test, y_test, len(x_test), 'SVM', 'SVM')
            p5.Plot_Resultados(10, 0, 1e-3, 32, 1e-2, x_train, y_train, x_test, y_test, len(x_test), 'SM', 'SM')
            plt.ylabel(r'Accuracy test')
            plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        plt.tight_layout()

#Copio datos de cifar10/mnist
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Preprocesado
x_train=np.int16(np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:]))))/255
y_train =np.reshape(y_train,(len(y_train)))
x_test=np.int16(np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:]))))/255
y_test =np.reshape(y_test,(len(y_test)))

yy_train=np.zeros((x_train.shape[0],10))
yy_train[np.arange(x_train.shape[0]),y_train]=1
yy_test=np.zeros((x_test.shape[0],10))
yy_test[np.arange(x_test.shape[0]),y_test]=1

x_train=x_train-np.mean(x_train,axis=0)
x_test=x_test-np.mean(x_train,axis=0)


b = np.ones((len(x_train), 1))
bt = np.ones((len(x_test), 1))
x_train = np.hstack((b,x_train))
x_test = np.hstack((bt,x_test))

w1=np.random.randn(len(x_train[0]),100)*1e-6
w2=np.random.randn(101,10)
lr=1e-6
bs=32
lambd=1e-5

#Fit

epoca=0
epocas_totales=20
loss=np.zeros(epocas_totales)
acc=np.zeros(epocas_totales)
acc_test=np.zeros(epocas_totales)
epocas=np.zeros(epocas_totales)

while epoca<epocas_totales:
    # Forward con test
    x_aux = scores(w1, x_test)
    s1 = sigmoid(x_aux)
    s1_aux = np.hstack((np.ones((len(x_test), 1)), s1))
    s2 = scores(w2, s1_aux)
    acc_test[epoca] = accuracy_rn(s2, yy_test)


    id_batch = np.arange(len(x_train))
    np.random.shuffle(id_batch)
#Forward
    for i in range(0, len(x_train), bs):
        #Selecciono el batch
        x_batch = x_train[id_batch[i: (i + bs)]]
        y_batch = yy_train[id_batch[i: (i + bs)]]
        #Primera capa
        x_aux=scores(w1,x_batch)
        s1=sigmoid(x_aux)
        s1_aux=np.hstack((np.ones((len(s1), 1)),s1))
        #Calculo de regularizacion para ambas capas
        reg1=np.sum(w1**2)
        reg2 = np.sum(w2 ** 2)
        reg=lambd*(reg1+reg2)
        #Segunda capa y calculo de loss/accuracy
        s2=scores(w2,s1_aux)
        loss[epoca]=loss[epoca]+MSE(s2,y_batch)+0.5*reg
        acc[epoca]=acc[epoca]+accuracy_rn(s2,y_batch)
    #Backward
        #Gradiente segunda capa
        grad=grad_MSE(s2,y_batch)
        gradW2=np.dot(s1_aux.T,grad)+w2*lambd
        grad=np.dot(grad,w2.T)
        grad=grad[:,1:]
        #Gradiente primera capa
        grad_sig=grad_sigmoid(s1)
        grad=grad*grad_sig
        gradW1=np.dot(x_batch.T,grad)+w1*lambd
        w1-=gradW1*lr
        w2-=gradW2*lr

    loss[epoca] /= int(len(x_train) / bs)
    acc[epoca] /= int(len(x_train) / bs)
    epocas[epoca] = epoca
    epoca += 1

plt.figure()
Plot_Resultados(epocas,loss,acc,acc_test,'Red neuronal de dos capas totalmente conectadas','resultados')
plt.show()
plt.figure()
Plot_Resultados(epocas,loss,acc,acc_test,'Red neuronal de dos capas totalmente conectadas','comparacion')
plt.show()

