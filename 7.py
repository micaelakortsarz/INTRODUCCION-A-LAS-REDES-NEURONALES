import numpy as np
import models
import activations
import optimizers
import layers
import matplotlib.pyplot as plt


def broadcasting_typecast(n):
    return -2*((np.arange(2**n)[:,None] & (1 << np.arange(n-1,-1,-1))) != 0)+1

def Plot_Resultados(epocas,loss,ac,label,N,M):
    with plt.style.context('seaborn-darkgrid'):

        plt.grid(True)

        ax1 = plt.subplot(211)
        plt.plot(epocas,loss, label=label)
     #   plt.title("N={}|N'={}".format(N, M))
        plt.ylabel(r'Loss')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        ax2 = plt.subplot(212)
        plt.plot(epocas,ac, label=label)
        plt.ylabel(r'Accuracy')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        plt.tight_layout()

m=[10,10,10,10]
n=[2,4,6,10]
lr=[1e-1,1e-1,8e-2,7e-3]
#n=[10,10,10,10,10,10]
#m=[2,4,6,8,10,50]
#lr=[1e-1,1e-1,1e-1,1e-1,1e-1,1e-2]
plt.figure()
for N,M,alpha in zip(n,m,lr):
    x_train = broadcasting_typecast(N)
    y_train = np.prod(x_train, axis=1)
    y_train = y_train.reshape((len(y_train), 1))
    test_data = [x_train, y_train]
    bs=len(x_train)
    lam=0.0
    opt=optimizers.BGD(alpha,bs,lam)

  #  np.random.seed(0)
    model=models.Network(opt)
    layer1=layers.Dense(M,activations.Tanh(),len(x_train[0]),bs)
    model.add(layer1)
    layer2=layers.Dense(1,activations.Tanh(),M,bs)
    model.add(layer2)
    ec,loss,acc,ac_t=model.fit(x_train,y_train,test_data=test_data,epochs=1000,bs=bs,lambd=lam,flag='Problema_XOR')
    Plot_Resultados(ec,loss,acc,"N={}|N'={}".format(N,M),N,M)
    #Plot_Resultados(ec, loss, acc, 'lr={}'.format(alpha),N,M)
plt.show()