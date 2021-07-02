import numpy as np

def MSE(scores,y_true):
    return np.mean(np.sum((scores-y_true)**2,axis=1))

def CCE(score,y_true):
    score -= np.max(score, axis=1)[:, np.newaxis]
    score_real = score[np.arange(len(score)), y_true]
    e_score = np.exp(score)
    e_score_sum = np.sum(e_score, axis=1)
    s = np.log(e_score_sum) - score_real
    return np.mean(s)

def regularizacion(w,lamb):
    return lamb*np.sum(w**2)*0.5

def accuracy(s,y_true):
    y_pred=np.argmax(s, axis=1)
    y_pred=y_pred.astype(np.int16)
    acc = (y_pred==y_true).mean()
    return acc

def accuracy_XOR(scores,y_true):
    scores[scores>0.9]=1
    scores[scores<-0.9]=-1
    acc = (scores==y_true).mean()
    return acc
