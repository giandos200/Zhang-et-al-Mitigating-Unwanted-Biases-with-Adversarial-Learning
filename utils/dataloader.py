import pandas as pd


def dataloader(data, sensitive_feature):
    if data.upper() =='GERMAN':
        df = pd.read_csv("data/German.tsv", sep='\t')
        numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'existingcredits', 'peopleliable']
        Sensitive_Features = ['gender', 'foreignworker']
        if sensitive_feature.lower() =='gender':
            target = df[['classification', Sensitive_Features[0]]]
            mappingPrivUnpriv = {'privilaged': 'M', 'unprivilaged': 'F'}
            target.replace(['M', 'F'], [1, 0], inplace=True)
            df = df.drop(columns=Sensitive_Features)
            dataset = df.drop("classification", axis=1)
            categorical = dataset.columns.difference(numvars)
            return (dataset,target,numvars,categorical)
        elif sensitive_feature.lower() == 'foreignworker':
            target = df[['classification', Sensitive_Features[1]]]
            mappingPrivUnpriv = {'privilaged': 'no', 'unprivilaged': 'yes'}
            target.replace(['no', 'yes'], [1, 0], inplace=True)
            df = df.drop(columns=Sensitive_Features)
            dataset = df.drop("classification", axis=1)
            categorical = dataset.columns.difference(numvars)
            return (dataset, target, numvars, categorical)
        else:
            if sensitive_feature not in df.columns.to_list():
                raise "The required Sensitive Feature does not exist"
            else:
                raise NotImplementedError


    elif data.upper() == 'ADULT':
        df = pd.read_csv("data/adult.tsv", sep='\t')
        df = df.dropna()
        numvars = ['education-num', 'capital gain', 'capital loss', 'hours per week','Age','fnlwgt']
        #******************************FOLLOWING, MY CONSIDERATION*****************************************************
        # to be fair the next features should be dropped since source of bias and collinearity, however in
        # the paper the n° of features are 14 so the authors make use of them
        #df = df.drop(columns=['Age', 'race', 'relationship', 'fnlwgt', 'education', 'native-country'])
        Sensitive_Features = ['gender', 'marital-status']
        if sensitive_feature.lower() == 'gender':
            target = df[["income", "gender"]]  # 'marital-status'
            target.replace([' Male', ' Female'], [1, 0], inplace=True)
            df = df.drop(columns=Sensitive_Features)
            dataset = df.drop("income", axis=1)
            categorical = dataset.columns.difference(numvars)
            return (dataset, target, numvars, categorical)
        elif sensitive_feature == 'marital-status':
            df.replace(to_replace=[' Divorced', ' Married-AF-spouse',
                                   ' Married-civ-spouse', ' Married-spouse-absent',
                                   ' Never-married', ' Separated', ' Widowed'], value=
                       ['not married', 'married', 'married', 'married',
                        'not married', 'not married', 'not married'], inplace=True)
            target = df[["income", "marital-status"]]
            target.replace(['married', 'not married'], [1, 0], inplace=True)
            df = df.drop(columns=Sensitive_Features)
            dataset = df.drop("income", axis=1)
            categorical = dataset.columns.difference(numvars)
            return (dataset, target, numvars, categorical)
        else:
            if sensitive_feature not in df.columns.to_list():
                raise "The required Sensitive Feature does not exist"
            else:
                raise NotImplementedError