from sklearn.metrics import confusion_matrix


def accuracy(y_true, y_pred):
    return 1-(abs(y_true - y_pred)).mean()

def discrimination(y_real,y_pred,SensitiveCat,privileged,unprivileged):
    y_priv = y_pred[y_real[SensitiveCat] == privileged]
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    return abs(y_priv.mean()-y_unpriv.mean())

def consistency(X,y_pred,k=5):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    y=0
    N = X.shape[0]
    for i in range(N):
        distances, indices = nbrs.kneighbors(X[i,:].reshape(1,-1))
        #THE FIRST INDEX IS ALWAYS THE SAME SAMPLE -> REMOVE [1:]\|/
        y+=abs(y_pred.iloc[i] - y_pred.iloc[indices.tolist()[0][1:]].sum())
    return 1-y/(N*k)

    pass

def DifferenceEqualOpportunity(y_pred,y_real,SensitiveCat, outcome, privileged, unprivileged, labels):
    '''
    ABS Difference in True positive Rate between the two groups
    :param y_pred: prediction
    :param y_real: real label
    :param SensitiveCat: Sensitive feature name
    :param outcome: Outcome feature name
    :param privileged: value of the privileged group
    :param unprivileged: value of the unprivileged group
    :param labels: both priv-unpriv value for CFmatrix
    :return:
    '''
    y_priv = y_pred[y_real[SensitiveCat]==privileged]
    y_real_priv = y_real[y_real[SensitiveCat]==privileged]
    y_unpriv = y_pred[y_real[SensitiveCat]==unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat]==unprivileged]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome],y_priv, labels=labels).ravel()
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv, labels=labels).ravel()

    return abs(TP_unpriv/y_real_unpriv.shape[0] - TP_priv/y_real_priv.shape[0])

def DifferenceAverageOdds(y_pred,y_real,SensitiveCat, outcome, privileged, unprivileged,labels):
    '''
    Mean ABS difference in True positive rate and False positive rate of the two groups
    :param y_pred:
    :param y_real:
    :param SensitiveCat:
    :param outcome:
    :param privileged:
    :param unprivileged:
    :param labels:
    :return:
    '''
    y_priv = y_pred[y_real[SensitiveCat] == privileged]
    y_real_priv = y_real[y_real[SensitiveCat] == privileged]
    y_unpriv = y_pred[y_real[SensitiveCat] == unprivileged]
    y_real_unpriv = y_real[y_real[SensitiveCat] == unprivileged]
    TN_priv, FP_priv, FN_priv, TP_priv = confusion_matrix(y_real_priv[outcome], y_priv,  labels=labels).ravel()
    TN_unpriv, FP_unpriv, FN_unpriv, TP_unpriv = confusion_matrix(y_real_unpriv[outcome], y_unpriv,  labels=labels).ravel()
    return 0.5*(abs(FP_unpriv/y_real_unpriv.shape[0]-FP_priv/y_real_priv.shape[0])+abs(TP_unpriv/y_real_unpriv.shape[0]-TP_priv/y_real_priv.shape[0]))
