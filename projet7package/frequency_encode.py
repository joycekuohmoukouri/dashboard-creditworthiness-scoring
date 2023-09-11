import pandas as pd

def frequency_encode(x):
    print("Reading data from 'df_train_set_1.csv'")
    df_train_org = pd.read_csv('df_train_set_1.csv', usecols=['SECTEUR_ACTIVITE'])
    freq_by_org_type = df_train_org['SECTEUR_ACTIVITE'].value_counts(normalize=True).to_dict()
    print("Frequency encoding calculations completed")
    return x.replace(freq_by_org_type)