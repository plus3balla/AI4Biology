import os
import pickle
import logging
import pathlib
import warnings

import numpy as np
from glob import glob
from preprocessing import read_data, prepocess_data
warnings.simplefilter('ignore')

logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


BASE_DIR = pathlib.Path(__file__).parent
PATH_TO_INPUT = '../input/'
PATH_TO_MODELS = BASE_DIR / 'models/'
PATH_TO_OUTPUT = BASE_DIR / 'output/'

if __name__ == '__main__':
    logger.info('Запуск программы')
    logger.info('Загрузка тестового датасета')
    test_df = read_data(os.path.join(PATH_TO_INPUT, 'test.json'))
    logger.info('Предобработка датасета')
    test_df = prepocess_data(test_df)
    logger.info(f'N примеров: {test_df.shape[0]}')

    X_pred = []
    for i in test_df.index:
        row = test_df.loc[i, 'intens']
        X_pred.append(row)
    X_pred = np.array(X_pred)

    model_names = glob(f'{PATH_TO_MODELS}/*.pkl')
    class_order = []
    prediction = []
    for model_name in model_names:
        with open(model_name, 'rb') as model_path:
            logger.info(f'Загрзука модели: {model_name}')
            strain = '_'.join(model_name.split('/')[-1].split('.')[0].split('_')[2:])
            class_order.append(strain)
            logger.info(f'Прогноз для штамма {strain}')
            model = pickle.load(model_path)
            prediction.append(model.predict_proba(X_pred)[:, 1])

    logger.info('Объединяем предикты')
    prediction = np.array(prediction).T

    class_name = []
    for p in prediction:
        if max(p) > 0.5:
            class_name.append(class_order[np.where(p == max(p))[0].item()])
        else:
            class_name.append('new')

    test_df.loc[:, 'target_class'] = class_name

    logger.info('Сохраняем предикты в results.csv')
    test_df[['target_class']].to_csv(os.path.join(PATH_TO_OUTPUT, 'results.csv'),
                                     index=True, index_label='id')
