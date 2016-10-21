from __future__ import print_function
from os.path import join
from os import environ
import click
import logging
import src.settings as settings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import importlib
from ast import literal_eval


def load_data_with_cv(dataset, cv_num):
    dataset_path = join(settings.project_dir, environ.get('PROCESSED_DATA_DIR'), dataset)
    data = pd.read_csv(join(dataset_path, 'data.csv'))
    y = np.load(join(dataset_path, 'train.{0}.npy'.format(cv_num)))
    y_t = np.load(join(dataset_path, 'test.{0}.npy'.format(cv_num)))
    return data.iloc[y, :], data.iloc[y_t, :]


def _transform(X,X_t,y,clf):
    clf.fit(X,y)
    return clf.predict(X),clf.predict(X_t)


@click.command()
@click.option('--dataset', type=click.STRING, help='The name of the dataset in the "data/processed" dir.')
@click.option('--cv/--no-cv', default=True, help="Use predifined Cross-Validation folds")
@click.option('--cv-num', type=click.INT, default=0, help="Use random split number #")
@click.option('--alg', default='BSP',
              type=click.Choice(['BSP','other']),
              help='The name of algorithm to do feature transformation. '
                   'The name should be either "BSP" or "other". if "other" then need '
                   '"--alg-package" and "--alg-method" to be set')
@click.option('--alg-package',type=click.STRING,default=None,help='the package root e.g "sklearn.svm"')
@click.option('--alg-method',type=click.STRING,default=None,help='the package method e.g "LinearSVC"')
@click.option('--alg-opts',default=None, type=click.STRING, help='A dict as a string of the options')
@click.option('--std', is_flag=True, default=True)
@click.option('--ezp-dim',tyoe=click.INT,default=1000)
@click.option('--ezp-rand', type=click.FLOAT,default=0.1)
@click.option('--p',is_flag=True)
def main(dataset, cv, cv_num, alg,alg_package,alg_method, alg_opts, std,ezp_dim,ezp_rand,p):
    logger = logging.getLogger(__name__)
    if cv:
        X, X_t = load_data_with_cv(dataset, cv_num=cv_num)
        y, y_t = X['class'], X_t['class']
        pred = [x for x in X.columns if x != 'class']
        if std:
            from sklearn.preprocessing import StandardScaler
            ss = StandardScaler()
            X.values = ss.fit(X[pred]).transform(X[pred])
            X_t.values = ss.transform(X_t[pred])
        if alg == 'BSP':
            pass
        elif alg == 'other' and alg_package and alg_method and alg_opts:
            opts = literal_eval(alg_opts)
            clf = getattr(importlib.import_module())
        else:
            logger.error('Error in options')
            logger.exception('the error happened here ')
            exit(1)

    # Until this part of the code data is ready and clf also is initialized
    # Now we check if "p" is set
    if p:
        pass
    else:
        sss = StratifiedShuffleSplit(n_splits=ezp_dim,train_size=ezp_rand,test_size=None)
        for i,tr_ind in enumerate(sss.split(X,y)):
            if i % 1000 == 0 and i is not 0:
                pass








if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
