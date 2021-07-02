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

def categorical_cross_entropy(score,y_true):
    score -= np.max(score, axis=1)[:, np.newaxis]
    score_real = score[np.arange(len(score)), y_true]
    e_score = np.exp(score)
    e_score_sum = np.sum(e_score, axis=1)
    s = np.log(e_score_sum) - score_real
    return np.mean(s)

def grad_categorical_cross_entropy(score,y_true):
    score -= np.max(score, axis=1)[:, np.newaxis]
    e_score = np.exp(score)
    e_score_sum = np.sum(e_score, axis=1)
    g = np.zeros(score.shape)
    g[np.arange(len(score)), y_true] = -1
    grad = g + (e_score / e_score_sum[:, np.newaxis])
    return grad

def accuracy(s,y_true):
    y_pred=np.argmax(s, axis=1)
    y_true=np.argmax(y_true, axis=1)
    acc = (y_pred==y_true).mean()
    return acc


def regularizacion(w1,w2,lambd):
    reg1 = np.sum(w1 ** 2)
    reg2 = np.sum(w2 ** 2)
    return lambd * (reg1 + reg2)

def forward_test(x_test,yy_test,w1,w2):
    x_aux = scores(w1, x_test)
    s1 = sigmoid(x_aux)
    s1_aux = np.hstack((np.ones((len(x_test), 1)), s1))
    s2 = scores(w2, s1_aux)
    return accuracy(s2, yy_test)

def red_neuronal(epocas_totales,metodo_loss,x_train,yy_train,y_train,x_test,yy_test,w1,w2):
    epoca = 0
    loss = np.zeros(epocas_totales)
    acc = np.zeros(epocas_totales)
    acc_test = np.zeros(epocas_totales)
    epocas = np.zeros(epocas_totales)
    lr = 1e-6
    bs = 32
    lambd = 1e-5
    while epoca < epocas_totales:
        acc_test[epoca] = forward_test(x_test, yy_test, w1, w2)

        id_batch = np.arange(len(x_train))
        np.random.shuffle(id_batch)
        # Forward
        for i in range(0, len(x_train), bs):
            # Selecciono el batch
            x_batch = x_train[id_batch[i: (i + bs)]]
            y_batch = yy_train[id_batch[i: (i + bs)]]
            # Primera capa
            x_aux = scores(w1, x_batch)
            s1 = sigmoid(x_aux)
            s1_aux = np.hstack((np.ones((len(s1), 1)), s1))

            # Calculo de regularizacion para ambas capas
            reg = regularizacion(w1, w2, lambd)
            # Segunda capa y calculo de loss/accuracy
            s2 = scores(w2, s1_aux)

            if metodo_loss == 'MSE':
                loss[epoca] = loss[epoca] + MSE(s2, y_batch) + 0.5 * reg
                grad = grad_MSE(s2, y_batch)
            if metodo_loss == 'CCE':
                loss[epoca] = loss[epoca] + categorical_cross_entropy(s2, y_train[id_batch[i: (i + bs)]]) + 0.5 * reg
                grad = grad_categorical_cross_entropy(s2, y_train[id_batch[i: (i + bs)]])

            acc[epoca] = acc[epoca] + accuracy(s2, y_batch)
            # Backward
            # Gradiente segunda capa
            gradW2 = np.dot(s1_aux.T, grad) + w2 * lambd
            grad = np.dot(grad, w2.T)
            grad = grad[:, 1:]
            # Gradiente primera capa
            grad_sig = grad_sigmoid(s1)
            grad = grad * grad_sig
            gradW1 = np.dot(x_batch.T, grad) + w1 * lambd
            w1 -= gradW1 * lr
            w2 -= gradW2 * lr

        loss[epoca] /= int(len(x_train) / bs)
        acc[epoca] /= int(len(x_train) / bs)
        epocas[epoca] = epoca
        epoca += 1
    return epocas,loss,acc,acc_test

def Plot_Resultados(epocas,loss,ac,ac_test,label):
    with plt.style.context('seaborn-darkgrid'):
        plt.grid(True)
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
        plt.legend()
        plt.tight_layout()

def Plot_Comparacion(epocas, ac_test_mse,ac_test_cce, label1,label2):
    with plt.style.context('seaborn-darkgrid'):
        plt.grid(True)
        plt.plot(epocas, ac_test_mse, 'o-', label=label1)
        plt.plot(epocas, ac_test_cce, 'o-', label=label2)
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

w1=np.random.randn(len(x_train[0]),100)*1e-3
w2=np.random.randn(101,10)
epocas_totales=20

epocas_mse,l_mse,ac_mse,ac_t_mse=red_neuronal(epocas_totales,'MSE',x_train,yy_train,y_train,x_test,yy_test,w1,w2)
w1=np.random.randn(len(x_train[0]),100)*1e-3
w2=np.random.randn(101,10)
epocas_cce,l_cce,ac_cce,ac_t_cce=red_neuronal(epocas_totales,'CCE',x_train,yy_train,y_train,x_test,yy_test,w1,w2)



plt.figure()
Plot_Resultados(epocas_mse,l_mse,ac_mse,ac_t_mse,'Mean square error (MSE)')
Plot_Resultados(epocas_cce,l_cce,ac_cce,ac_t_cce,'Categorical cross entropy (CCE)')
plt.show()


plt.figure()
Plot_Comparacion(epocas_mse, ac_t_mse, ac_t_cce, 'Mean square error', 'Categorical cross entropy')
plt.show()


