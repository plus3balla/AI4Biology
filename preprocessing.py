import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder


def read_data(f):
    with open(f, 'rb') as fp:
        test_d = json.load(fp)
        test_df = pd.DataFrame(json.loads(test_d)).T
    return test_df


def create_spectre(mz, intens):
    spec = []
    for i in range(200, 1750):
        if i in mz:
            spec.append(intens[mz.index(i)])
        else:
            spec.append(0)
    return spec


def preprocess_data(data):
    data['mz'] = data['m/z'].apply(lambda x: [int(x_i // 10) for x_i in x])
    data['intens'] = data.apply(lambda d: create_spectre(d['mz'], d['Rel. Intens.']),
                                axis=1)
    return data


def encode_strains(data: pd.DataFrame):
    data['strain_encoded'] = LabelEncoder().fit_transform(data['strain'])
    temp = data[['strain_encoded', 'strain']].drop_duplicates()

    decoder = temp.set_index('strain_encoded').to_dict()['strin']

    return decoder
