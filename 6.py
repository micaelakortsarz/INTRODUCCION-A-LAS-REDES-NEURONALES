import numpy as np
import models
import activations
import optimizers
import layers
import matplotlib.pyplot as plt
np.random.seed(30)
def Plot_Resultados(epocas,loss,ac,label):
    with plt.style.context('seaborn-darkgrid'):
        plt.grid(True)

        ax1 = plt.subplot(211)
        plt.plot(epocas,loss, label=label)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        ax2 = plt.subplot(212)
        plt.plot(epocas,ac, label=label)
        plt.ylabel(r'Accuracy')
        plt.xlabel(r'Epocas transcurridas')
        plt.legend()
        plt.tight_layout()

        plt.legend()
        plt.tight_layout()



x_train=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
y_train=np.array([[1],[-1],[-1],[1]])
test_data=[x_train,y_train]
bs=4
lambd=0
opt=optimizers.BGD(0.05,bs,lambd)

model=models.Network(opt)
layer1=layers.Dense(2,activations.Tanh(),len(x_train[0]),bs)
model.add(layer1)
layer2=layers.Dense(1,activations.Tanh(),len(x_train[0]),bs)
model.add(layer2)
ec,loss,acc,ac_t=model.fit(x_train,y_train,test_data=test_data,bs=bs,lambd=lambd)
plt.figure()
Plot_Resultados(ec,loss,acc,'Arquitectura 1')


model2=models.Network(opt)
layer1_2=layers.Dense(1,activations.Tanh(),len(x_train[0]),bs)
input=layer1_2(x_train)

layer2_2=layers.ConcatInput()
aux=layer2_2(input,x_train)

layer3_2=layers.Dense(1,activations.Tanh(),len(aux[0]),bs)
model2.add(layer1_2)
model2.add(layer2_2)
model2.add(layer3_2)
ec2,loss2,acc2,ac2_t =model2.fit(x_train,y_train, test_data=test_data,bs=bs,lambd=lambd)


Plot_Resultados(ec2,loss2,acc2,'Arquitectura 2')
plt.show()
