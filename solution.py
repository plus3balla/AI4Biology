import os
import pickle
import logging
import pathlib
import warnings

from tensorflow.keras.models import load_model
import numpy as np
from preprocessing import read_data, preprocess_data
from glob import glob
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
    test_df = preprocess_data(test_df)
    logger.info('Кодирование штаммов')
    decoder = pickle.load(open("models/decoder.pkl", "rb"))
    logger.info(f'N примеров: {test_df.shape[0]}')

    X_pred = []
    for i in test_df.index:
        row = test_df.loc[i, 'intens']
        X_pred.append(row)
    X_pred = np.array(X_pred)

    model = load_model("models/nn")
    prediction = model.predict(X_pred)

    class_name = []
    for p in prediction:
        if max(p) > 1.0:
            class_name.append(decoder[p.argmax()])
        else:
            class_name.append('new')

    test_df.loc[:, 'target_class'] = class_name

    logger.info('Сохраняем предикты в results.csv')
    test_df[['target_class']].to_csv(os.path.join(PATH_TO_OUTPUT, 'results.csv'),
                                     index=True, index_label='id')
