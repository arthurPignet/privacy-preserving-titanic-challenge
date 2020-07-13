import pandas as pd

from sklearn.preprocessing import LabelEncoder

DATA_PATH = "../../data/raw/"
WRITE_PATH = "../../data/processed/"


def data_import(path=DATA_PATH):
    """"
    This function import the raw data from .csv files to pandas. It aims for 2 files, named train.csv and test.csv

    Parameter
    -----------
    path : path of the directory where the two .csv files are stored

    Returns
    ------------
    tuple : (pandas_df: train_set, pandas_df: test_set)
    """
    raw_train_set_df = pd.read_csv(path + "train.csv")
    raw_test_set_df = pd.read_csv(path + "test.csv")
    return raw_train_set_df, raw_test_set_df


def processing(raw_train_df, raw_test_df, isSaved=False, write_path=WRITE_PATH):
    """
    process the raw data, filling empty values, generating some features

    For commentary about this processing,
    see the notebook entitled processing in the directory
    titanic-data

    :parameter
    ------------
    train_df : dataframe with train_df data
    preprocessed_test_df : dataframe with test data
    (Optional) isSaved : boolean, if True the output data will be stored in the write_path
    (Optional) write_path : if isSaved, path where the output dataframe will be saved, in csv format.
    :returns
    ----------

    return
    """
    data_df = pd.concat([raw_train_df, raw_test_df], sort=True).reset_index(drop=True)
    data_df.Embarked = data_df.Embarked.fillna('S')
    data_df["Age"] = data_df.groupby(['Sex', 'Pclass', 'Embarked'])["Age"].apply(lambda x: x.fillna(x.median()))
    data_df.Fare = data_df.Fare.fillna(data_df.groupby(['Pclass', 'Parch']).median().Fare[3][0])
    data_df['Deck'] = data_df.Cabin.fillna('M').apply(lambda x: str(x)[0])
    data_df['Family_Size'] = data_df['SibSp'] + data_df['Parch'] + 1
    data_df['Title'] = data_df.Name.str.split(',', expand=True)[1].str.split('.', expand=True)[0].str.replace(" ", "")

    data_df['Title'] = data_df['Title'].replace(
        ['Lady', 'theCountess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data_df['Title'] = data_df['Title'].replace('Mlle', 'Miss')
    data_df['Title'] = data_df['Title'].replace('Ms', 'Miss')
    data_df['Title'] = data_df['Title'].replace('Mme', 'Mrs')

    categorical_col = ["Pclass", 'Embarked', 'SibSp', 'Deck', "Title"]

    data_df.Sex = LabelEncoder.fit_transform(data_df.Sex, data_df.Sex)

    for col in categorical_col:
        dummies = pd.get_dummies(data_df[col], prefix=col)
        data_df = pd.concat([data_df, dummies], axis=1)
        data_df = data_df.drop(col, axis=1)

    data_df.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis='columns', inplace=True)
    col_to_reg = ['Age', 'Fare', 'Family_Size']
    for col in col_to_reg:
        data_df[col] = (data_df[col] - data_df[col].mean()) / data_df[col].std()

    preprocessed_train_df = data_df.iloc[:raw_train_df.shape[0]]
    preprocessed_test_df = data_df.iloc[raw_train_df.shape[0]:].drop('Survived', axis=1)

    if isSaved:
        preprocessed_train_df.to_csv(write_path + "preprocessed_train.csv")
        preprocessed_test_df.to_csv(write_path + "preprocessed_test.csv")
    return preprocessed_train_df, preprocessed_test_df


if __name__ == "__main__":
    train_df, test_df = data_import()
    processing(train_df, test_df, isSaved=True)
