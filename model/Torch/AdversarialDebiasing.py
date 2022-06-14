import os
import random
import numpy as np
import scipy.special
import scipy
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset



class classifier_model(nn.Module):
    def __init__(self,feature,Hneuron1,output,dropout,seed1,seed2):
        super(classifier_model, self).__init__()
        self.feature = feature
        self.hN1 = Hneuron1
        self.output = output
        self.dropout = dropout
        self.seed1 = seed1
        self.seed2 = seed2
        self.FC1 = nn.Linear(self.feature,self.hN1)
        self.FC2 = nn.Linear(self.hN1,self.output)
        self.sigmoid = torch.sigmoid
        self.relu = F.relu
        self.Dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):
        x = self.Dropout(self.relu(self.FC1(x)))
        x_logits = self.FC2(x)
        x_pred = self.sigmoid(x_logits)
        return x_pred, x_logits

class adversary_model(nn.Module):
    def __init__(self, seed3, n_groups=1):
        super(adversary_model, self).__init__()
        self.seed3 = seed3
        self.c = torch.FloatTensor([1.0])
        self.FC1 = nn.Linear(3,n_groups)
        self.sigmoid = torch.sigmoid


    def forward(self,pred_logits, true_labels):
        s = self.sigmoid((1+torch.abs(self.c.to(pred_logits.device))) * pred_logits)
        pred_protected_attribute_logits = self.FC1(torch.cat([s, s * true_labels, s * (1.0 - true_labels)],1))
        pred_protected_attribute_labels = self.sigmoid(pred_protected_attribute_logits)
        return pred_protected_attribute_labels, pred_protected_attribute_logits




class AdversarialDebiasing(BaseEstimator, ClassifierMixin):
    """Debiasing with adversarial learning.

    'Torch implementation of AIF360.adversarialdebiasing and fairer reproduction
    of Zhang et al. work.'

    Adversarial debiasing is an in-processing technique that learns a
    classifier to maximize prediction accuracy and simultaneously reduce an
    adversary's ability to determine the protected attribute from the
    predictions [#zhang18]_. This approach leads to a fair classifier as the
    predictions cannot carry any group discrimination information that the
    adversary can exploit.

    References:
        .. [#zhang18] `B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating
           Unwanted Biases with Adversarial Learning," AAAI/ACM Conference on
           Artificial Intelligence, Ethics, and Society, 2018.
           <https://dl.acm.org/citation.cfm?id=3278779>`_

    Attributes:
        prot_attr_ (str or list(str)): Protected attribute(s) used for
            debiasing.
        groups_ (array, shape (n_groups,)): A list of group labels known to the
            classifier.
        classes_ (array, shape (n_classes,)): A list of class labels known to
            the classifier.
        sess_ (tensorflow.Session): The TensorFlow Session used for the
            computations. Note: this can be manually closed to free up resources
            with `self.sess_.close()`.
        classifier_logits_ (tensorflow.Tensor): Tensor containing output logits
            from the classifier.
        adversary_logits_ (tensorflow.Tensor): Tensor containing output logits
            from the adversary.
    """

    def __init__(self, prot_attr=None, scope_name='classifier',
                 adversary_loss_weight=0.1, num_epochs=50, batch_size=256,
                 classifier_num_hidden_units=200, debias=True, verbose=False,
                 random_state=None):
        r"""
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use in the debiasing process. If more than one
                attribute, all combinations of values (intersections) are
                considered. Default is ``None`` meaning all protected attributes
                from the dataset are used.
            scope_name (str, optional): TensorFlow "variable_scope" name for the
                entire model (classifier and adversary).
            adversary_loss_weight (float or ``None``, optional): If ``None``,
                this will use the suggestion from the paper:
                :math:`\alpha = \sqrt(global_step)` with inverse time decay on
                the learning rate. Otherwise, it uses the provided coefficient
                with exponential learning rate decay.
            num_epochs (int, optional): Number of epochs for which to train.
            batch_size (int, optional): Size of mini-batch for training.
            classifier_num_hidden_units (int, optional): Number of hidden units
                in the classifier.
            debias (bool, optional): If ``False``, learn a classifier without an
                adversary.
            verbose (bool, optional): If ``True``, print losses every 200 steps.
            random_state (int or numpy.RandomState, optional): Seed of pseudo-
                random number generator for shuffling data and seeding weights.
        """

        self.prot_attr = prot_attr
        self.scope_name = scope_name
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.debias = debias
        self.verbose = verbose
        self.random_state = random_state
        self.features_dim = None
        self.features_ph = None
        self.protected_attributes_ph = None
        self.true_labels_ph = None
        self.pred_labels = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_all_seed(self, seed):
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def init_parameters(self, net):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform(m.weight.data)
                torch.nn.init.normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

    def fit(self, X, y):
        """Train the classifier and adversary (if ``debias == True``) with the
        given training data.

        Args:
            X (pandas.DataFrame): Training samples.
            y (array-like): Training labels.

        Returns:
            self
        """
        if scipy.sparse.issparse(X):
            X=X.todense()
        X = torch.tensor(X.astype(np.float32)).to(self.device)
        groups = y[self.prot_attr]
        ys = torch.tensor(groups.values.astype(np.float32)).to(self.device)
        y_clf = y.drop(columns=self.prot_attr)
        y = torch.tensor(y_clf.values.astype(np.float32)).to(self.device)
        rng = check_random_state(self.random_state)
        if self.random_state is not None:
            self.set_all_seed(self.random_state)
        else:
            self.set_all_seed(42)
        ii32 = np.iinfo(np.int32)
        self.s1, self.s2, self.s3 = rng.randint(ii32.min, ii32.max, size=3)

        #groups, self.prot_attr_ = check_groups(X, self.prot_attr)
        # use sigmoid for binary case
        n_classes = y_clf.nunique().item()
        self.classes_ = np.unique(y_clf)
        n_groups = groups.nunique().item()
        if n_classes == 2:
            n_classes = 1
        if n_groups == 2:
            n_groups = 1
        if n_groups>2:
            # For intersection of more sensitive variable, i think this way
            # is not corrected! maybe should be prefered to build n different adversary model
            # instead of a unique handling both the sensitive classes
            OHE = OneHotEncoder()
            groups = OHE.fit_transform(groups.reshape(-1,1)).toarray()

        num_train_samples, n_features = X.shape

        starter_learning_rate = 0.001
        self.clf_model = classifier_model(feature=n_features, Hneuron1=self.classifier_num_hidden_units,
                                          output=n_classes, dropout=0.2,
                                          seed1=self.s1, seed2=self.s2).to(self.device)
        self.init_parameters(self.clf_model)
        classifier_opt = torch.optim.Adam(self.clf_model.parameters(),lr=starter_learning_rate,weight_decay=1e-5)
        # decayRate = 0.96
        # clf_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=classifier_opt, gamma=decayRate)
        # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(starter_learning_rate,
        #                                                                decay_steps=1000, decay_rate=0.96,
        #                                                                staircase=True)
        # classifier_opt = tf.optimizers.Adam(learning_rate)
        # classifier_vars = [var for var in self.clf_model.trainable_variables]
        dataBatch = DataLoader(TensorDataset(X, y, ys), batch_size=self.batch_size, shuffle=True,
                               drop_last=False)
        lossAdv = lossCLF = F.binary_cross_entropy_with_logits
        # pretrain_both_models
        if self.debias:
            self.adv_model = adversary_model(seed3=self.s3, n_groups=n_groups).to(self.device)
            self.init_parameters(self.adv_model)
            adversary_opt = torch.optim.Adam(self.adv_model.parameters(), lr=starter_learning_rate, weight_decay=1e-5)
            # decayRate = 0.96
            # adv_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=adversary_opt, gamma=decayRate)
            with tqdm(range(self.num_epochs//2)) as epochs:
                epochs.set_description("Classifcation PreTraining Epoch")
                for epoch in epochs:
                    self.clf_model.train()
                    for X_b, y_b, ys_b in dataBatch:
                        classifier_opt.zero_grad()
                        pred_labels, pred_logits = self.clf_model.forward(X_b)
                        loss = lossCLF(pred_logits,y_b,reduction='mean')
                        loss.backward()
                        classifier_opt.step()

                        # clf_lr_scheduler.step()

                        acc_b = (pred_labels.round()==y_b).float().sum().item()/X_b.size(0)
                        epochs.set_postfix(loss=loss.item(),acc=acc_b)

            with tqdm(range(10)) as epochs:
                epochs.set_description("Adversarial PreTraining Epoch")
                for epoch in epochs:
                    self.adv_model.train()
                    self.clf_model.eval()
                    for X_b, y_b, ys_b in dataBatch:
                        adversary_opt.zero_grad()
                        pred_labels, pred_logits = self.clf_model.forward(X_b)
                        pred_protected_attributes_labels, pred_protected_attributes_logits = self.adv_model.forward(
                            pred_logits, y_b)
                        loss = lossAdv(pred_protected_attributes_logits, ys_b, reduction='mean')
                        loss.backward()
                        adversary_opt.step()
                        # adv_lr_scheduler.step()

                        acc_b = (pred_protected_attributes_labels.round() == ys_b).float().sum().item()/X_b.size(0)
                        epochs.set_postfix(loss=loss.item(), acc=acc_b)

            print('\\Starting Debiasing Mitigation...\\')
            # clf_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=classifier_opt, gamma=decayRate)
            # adv_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=adversary_opt, gamma=decayRate)
            with tqdm(range(self.num_epochs)) as epochs:
                epochs.set_description("Adversarial Debiasing Training Epoch")
                for epoch in epochs:
                    self.adv_model.train()
                    self.clf_model.train()
                    for X_b, y_b, ys_b in dataBatch:
                        classifier_opt.zero_grad()
                        adversary_opt.zero_grad()
                        pred_labels, pred_logits = self.clf_model.forward(X_b)
                        lossclf1 = lossCLF(pred_logits,y_b,reduction='mean')
                        lossclf1.backward(retain_graph=True)
                        #dW_LP
                        clf_grad = [torch.clone(par.grad.detach()) for par in self.clf_model.parameters()]

                        classifier_opt.zero_grad()
                        adversary_opt.zero_grad()

                        pred_protected_attributes_labels, pred_protected_attributes_logits = self.adv_model.forward(
                            pred_logits, y_b)

                        lossadv1 = lossAdv(pred_protected_attributes_logits, ys_b, reduction='mean')
                        lossadv1.backward()
                        #dW_LA
                        adv_grad = [
                            torch.clone(par.grad.detach()) for par in self.clf_model.parameters()
                        ]

                        for i,par in enumerate(self.clf_model.parameters()):
                            # Normalization
                            unit_adversary_grad = adv_grad[i] / (torch.norm(adv_grad[i]) + torch.finfo(float).tiny)
                            # projection proj_{dW_LA}(dW_LP)
                            proj = torch.sum(torch.inner(unit_adversary_grad,clf_grad[i]))
                            # integrating into the CLF gradient
                            par.grad = clf_grad[i] - (proj * unit_adversary_grad) - (self.adversary_loss_weight * adv_grad[i])

                        classifier_opt.step()
                        # optimizing dU_LA
                        adversary_opt.step()
                        # clf_lr_scheduler.step()
                        # adv_lr_scheduler.step()
                        acc_adv = (pred_protected_attributes_labels.round() == ys_b).float().sum().item() / X_b.size(0)
                        acc_clf = (pred_labels.round() == y_b).float().sum().item() / X_b.size(0)
                        epochs.set_postfix(lossCLF=lossclf1.item(), lossAdv=lossadv1.item(), accCLF = acc_clf, accADV=acc_adv)
        else:
            with tqdm(range(self.num_epochs)) as epochs:
                epochs.set_description("Classifier Training Epoch")
                for epoch in epochs:
                    self.clf_model.train()
                    for X_b, y_b, ys_b in dataBatch:
                        classifier_opt.zero_grad()
                        pred_labels, pred_logits = self.clf_model.forward(X_b)
                        loss = lossCLF(pred_logits, y_b, reduction='mean')
                        loss.backward()
                        classifier_opt.step()
                        # clf_lr_scheduler.step()

                        acc_b = (pred_labels.round() == y_b).float().sum().item() / X_b.size(0)
                        epochs.set_postfix(loss=loss.item(), acc=acc_b)
        return self

    def decision_function(self, X):
        """Soft prediction scores.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Confidence scores per (sample, class) combination. In
            the binary case, confidence score for ``self.classes_[1]`` where >0
            means this class would be predicted.
        """
        if scipy.sparse.issparse(X):
            X = X.todense()
        X = torch.tensor(X.astype(np.float32)).to(self.device)
        #n_classes = len(self.classes_)

        #if n_classes == 2:
        #    n_classes = 1 # lgtm [py/unused-local-variable]

        # self.clf_model.eval()
        pred_labels_list = []
        dataBatch = DataLoader(X, batch_size=self.batch_size, shuffle=False,
                               drop_last=False)
        for X_b in dataBatch:
            self.clf_model.eval()
            pred_labels, pred_logits = self.clf_model.forward(X_b)
            pred_labels_list += pred_labels.cpu().detach().numpy().tolist()

        scores = np.array(pred_labels_list, dtype=np.float64).reshape(-1, 1)
        return scores.ravel() if scores.shape[1] == 1 else scores





    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the label of
        classes.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Returns the probability of the sample for each class
            in the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        decision = self.decision_function(X)

        if decision.ndim == 1:
            decision_2d = np.c_[np.zeros_like(decision), decision]
        else:
            decision_2d = decision
        return scipy.special.softmax(decision_2d, axis=1)

    def predict(self, X):
        """Predict class labels for the given samples.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if scores.ndim == 1:
            if X.shape[0]==1:
                indices = (scores > 0.5).astype(np.int).reshape((-1,))
            else:
                indices = (scores > 0.5).astype(np.int).reshape((-1,))
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]