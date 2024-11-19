from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

from utils.metrics import *
from utils.dataloader import dataloader
# from model.TF2.AdversarialDebiasing import AdversarialDebiasing as AdversarialDebiasingTF2
# from model.Torch.AdversarialDebiasing import AdversarialDebiasing as AdversarialDebiasingTorch
# import matplotlib.pyplot as plt
import time
import sys


def train_PIPELINE(DF,sensitive_feature,Backend,debiased=True):
    data = dataloader(DF, sensitive_feature=sensitive_feature)  # else adult
    n_epoch = 100 if DF=='adult' else 500
    dataset, target, numvars, categorical = data
    # Split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(dataset,
                                                        target,
                                                        test_size=0.1,
                                                        random_state=42,
                                                        stratify=target)
    classification = target.columns.to_list()
    classification.remove(sensitive_feature)
    classification = classification[0]
    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numvars),
            ('cat', categorical_transformer, categorical)])

    if Backend.lower() == 'torch':
        from model.Torch.AdversarialDebiasing import AdversarialDebiasing
    elif Backend.lower() in ['tf','tf2']:
        from model.TF2.AdversarialDebiasing import AdversarialDebiasing
    else:
        raise ValueError
    clf = AdversarialDebiasing([sensitive_feature], adversary_loss_weight=0.1, num_epochs=n_epoch, batch_size=256,
                                 classifier_num_hidden_units=256, random_state=42,debias=debiased)
    pipeline = Pipeline(steps=[('preprocessor', transformations),
                               ('classifier', clf)])
    start = time.time()
    pipeline.fit(x_train,y_train)
    end = time.time()
    total = end - start
    if debiased:
        CLF_type = 'Adversarial Debiasing'
    else:
        CLF_type = 'Biased Classification'
    print(f"{CLF_type} with {Backend} Backend Training completed in {total} seconds!")


    print("\nTrain Results\n")
    y_pred = pipeline.predict(x_train)
    ACC = accuracy_score(y_train[classification], y_pred)
    DEO = DifferenceEqualOpportunity(y_pred, y_train, sensitive_feature, classification, 1, 0, [0, 1])
    DAO = DifferenceAverageOdds(y_pred, y_train, sensitive_feature, classification, 1, 0, [0, 1])
    print(f'\nTrain Acc: {ACC}, \nDiff. Equal Opportunity: {DEO}, \nDiff. in Average Odds: {DAO}')

    start = time.time()
    print("\nTest Results\n")
    y_pred = pipeline.predict(x_test)
    ACC = accuracy_score(y_test[classification], y_pred)
    DEO = DifferenceEqualOpportunity(y_pred, y_test, sensitive_feature, classification, 1, 0, [0, 1])
    DAO = DifferenceAverageOdds(y_pred, y_test, sensitive_feature, classification, 1, 0, [0, 1])
    print(f'\nTest Acc: {ACC}, \nDiff. Equal Opportunity: {DEO}, \nDiff. in Average Odds: {DAO}')
    end = time.time()
    total = end - start
    print(f"{CLF_type} with {Backend} Backend Inference completed in {total} seconds!")


if __name__=='__main__':
    info = sys.argv[1].split('_')
    dataframe = info[0]
    sensitive_feature = info[1]
    Backend = info[2]

    train_PIPELINE(
        DF=dataframe,
        sensitive_feature=sensitive_feature,
        Backend=Backend,
        debiased=False
    )
    train_PIPELINE(
        DF=dataframe,
        sensitive_feature=sensitive_feature,
        Backend=Backend,
        debiased=True
    )
