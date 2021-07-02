import layers
import metrics
import losses
import numpy as np

class Network():
    def __init__(self,opt):
        self.layers=[]
        self.opt=opt

    def add(self,layer):
        if isinstance(layer,layers.BaseLayer)==False:
            print("Error:No es un objeto capa")
            return
        if len(self.layers)>0 and isinstance(layer,layers.WLayer)==True:
            self.layers.append(layer)
            self.layers[-1].set_input_shape(self.layers[-2].get_output_shape())
            self.layers[-1].input_shape[1]+=1
            self.layers[-1].init_weights()

        if len(self.layers)==0 and isinstance(layer,layers.WLayer)==True:
            if layer.get_input_shape()== [None,None]:
                print("Error:Input incorrecto")
                return
            else:
                self.layers.append(layer)
                self.layers[-1].init_weights()

        if isinstance(layer, layers.ConcatInput) == True:
            self.layers.append(layer)


    def get_layer(self,number_of_layer):
        return self.layers[number_of_layer]

    def forward_upto(self,j,x):
        x_aux=x
        self.scores=[]
        self.sy=[]
        self.primados=[]
        for c in self.layers:
            if isinstance(c, layers.ConcatInput) == True:
                s=c(x_aux,x)
                sy=s
            else:
                s=c(x_aux)
                sy=c.producto(x_aux)

            self.primados.append(c.build_bias(x_aux))
            self.sy.append(sy)
            self.scores.append(s)
            x_aux=s
        return self.scores[-1]

    def backward(self,grad_loss,x):
        grad=grad_loss
        for i in reversed(range(len(self.layers))):
            if isinstance(self.layers[i], layers.ConcatInput) == True:
                grad = grad[:, 0:len(self.scores[i-1][1])]
                continue
            if isinstance(self.layers[i], layers.WLayer) == True:
                grad_aux=self.layers[i].activation.gradient(self.sy[i])
                grad=grad_aux*grad
                aux=np.dot(self.primados[i].T,grad)
                self.layers[i].update_weights(self.opt,aux)
                grad=np.dot(grad,self.layers[i].W.T)
                grad = grad[:, 1:]



    def predict(self,x):
        scores=self.forward_upto(len(self.layers),x)
        y_pred=np.zeros_like(scores)
        y_pred[np.arange(len(scores)),np.argmax(scores)]=1
        return y_pred

    def predict_XOR(self,x,y):
        scores = self.forward_upto(len(self.layers), x,y)
        scores[scores > 0.9] = 1
        scores[scores < -0.9] = -1
        return scores

    def regularizacion(self,lambd):
        n = len(self.layers)
        aux=0
        for i in range(n):
            if isinstance(self.layers[i], layers.WLayer) == True:
                aux+=metrics.regularizacion(self.layers[i].W, lambd)
        return aux

    def fit(self,x,y,test_data=[None,None],epochs=500,bs=200,lambd=1e-4,loss_fun=losses.MSE(),flag='Problema_XOR'):
        epoch=0
        loss = np.zeros(epochs)
        ac = np.zeros(epochs)
        ac_t = np.zeros(epochs)
        ec=np.zeros(epochs)
        n=len(self.layers)
        N=int(len(x) / bs)
        while epoch<epochs:
            if len(test_data[0])!=0:
                scores = self.forward_upto(n, test_data[0])
                ac_t[epoch]=metrics.accuracy(scores,test_data[1])

            id_batch = np.arange(len(x))
            np.random.shuffle(id_batch)
            for i in range(0, len(x), bs):
                # Selecciono el batch
                x_batch = x[id_batch[i: (i + bs)]]
                y_batch = y[id_batch[i: (i + bs)]]
                scores=self.forward_upto(n,x_batch)
                loss[epoch]+=loss_fun(scores,y_batch,flag)+self.regularizacion(lambd)
                if flag=='Problema_XOR':
                    ac[epoch]+=metrics.accuracy_XOR(scores,y_batch)
                else: ac[epoch]+=metrics.accuracy(scores,y_batch)
                grad_loss=loss_fun.gradient(scores,y_batch,flag)
                self.backward(grad_loss,x_batch)
            loss[epoch] /= N
            ac[epoch] /= N
            print(ac[epoch],loss[epoch],ac_t[epoch])
            ec[epoch]=epoch
            epoch+=1
        print('Transcurrieron {} epocas'.format(epoch))
        return ec,loss,ac,ac_t