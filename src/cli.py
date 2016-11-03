# Python imports
from __future__ import print_function
from os.path import join
from os import environ
import logging
import importlib
from ast import literal_eval

# 3d Party imports
import click
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
# Package imports
import settings
import features.build_features as feat_trans
import models.train_model as train


def load_data_with_cv(dataset, cv_num):
    dataset_path = join(settings.project_dir, environ.get('PROCESSED_DATA_DIR'), dataset)
    data = pd.read_csv(join(dataset_path, 'data.csv'))
    y = np.load(join(dataset_path, 'train.{0}.npy'.format(cv_num)))
    y_t = np.load(join(dataset_path, 'test.{0}.npy'.format(cv_num)))
    return data.iloc[y, :], data.iloc[y_t, :]



@click.group()
@click.option('--dataset', type=click.STRING, help='The name of the dataset in the "data/processed" dir.')
@click.option('--cv/--no-cv', default=True, help="Use predifined Cross-Validation folds")
@click.option('--cv-num', type=click.INT, default=0, help="Use random split number #")
@click.option('--std', is_flag=True, default=True)
@click.pass_context
def main(ctx,dataset, cv, cv_num,std):
    logger = logging.getLogger(__name__)
    if cv:
        X, X_t = load_data_with_cv(dataset, cv_num=cv_num)
        pred = [x for x in X.columns if x != 'class']
        if std:
            from sklearn.preprocessing import StandardScaler
            ss = StandardScaler()
            X[pred] = ss.fit(X[pred]).transform(X[pred])
            X_t[pred] = ss.transform(X_t[pred])

    logger.info('Dataset : {0} with rows {1}, cols {2}, Num of classes {3}'.format(dataset,X.shape[0]+X_t.shape[0],
                                                                                   X.shape[1],
                                                                                   np.unique(X['class']).shape[0]))
    ctx.obj['dataset'] = dataset
    ctx.obj['cv_num'] = cv_num
    ctx.obj['X'] = X
    ctx.obj['X_t'] = X_t


@main.command()
@click.option('--alg', default='BSP',
              type=click.Choice(['BSP', 'other']),
              help='The name of algorithm to do feature transformation. '
                   'The name should be either "BSP" or "other". if "other" then need '
                   '"--alg-package" and "--alg-method" to be set')
@click.option('--alg-pkg', type=click.STRING, default=None, help='the package root e.g "sklearn.svm"')
@click.option('--alg-f', type=click.STRING, default=None, help='the package method e.g "LinearSVC"')
@click.option('--alg-o', default=None, type=click.STRING, help='A dict as a string of the options')
@click.option('--ezp-dim', type=click.INT, default=1000)
@click.option('--ezp-rand', type=click.FLOAT, default=0.1)
@click.option('--p', is_flag=True)
@click.pass_context
def build_feat(ctx,alg, alg_pkg, alg_f, alg_o, ezp_dim, ezp_rand, p):
    logger = logging.getLogger(__name__)
    X = ctx.obj['X']
    X_t = ctx.obj['X_t']

    if alg == 'BSP':
        pass
    elif alg == 'other' and alg_pkg and alg_f and alg_o:
        opts = literal_eval(alg_o)
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
        sss = StratifiedShuffleSplit(n_splits=ezp_dim, train_size=ezp_rand, test_size=None)
        # for i, tr_ind in enumerate(sss.split(X, y)):
        #     if i % 1000 == 0 and i is not 0:
        #         pass


@main.command()
@click.option('--feat/--no-feat',default=False,help='Did you use Build_Feat before?')
@click.option('--alg', default='BSP',
              type=click.Choice(['BSP', 'other']),
              help='The name of algorithm to do feature transformation. '
                   'The name should be either "BSP" or "other". if "other" then need '
                   '"--alg-package" and "--alg-method" to be set')
@click.option('--alg-pkg', type=click.STRING, default=None, help='the package root e.g "sklearn.svm"')
@click.option('--alg-f', type=click.STRING, default=None, help='the package method e.g "LinearSVC"')
@click.option('--alg-o', default=None, type=click.STRING, help='A dict as a string of the options')
@click.option('--alg-cv',default=None,type=click.STRING,help='A dict as a string of the options to be cross-validated')
@click.pass_context
def train_test(ctx,feat,alg,alg_pkg, alg_f, alg_o,alg_cv):
    logger = logging.getLogger(__name__)
    if feat:
        pass
    else:
        logger.info('Using Original data...')
        X = ctx.obj['X']
        X_t = ctx.obj['X_t']

    if alg == 'BSP':
        pass
    elif alg == 'other' and alg_pkg and alg_f and alg_o:
        opts = literal_eval(alg_o)
        clf = getattr(importlib.import_module(alg_pkg),alg_f)()
        clf.set_params(**opts)
    else:
        logger.error('Error in options')
        logger.exception('the error happened here ')
        exit(1)
    if alg_cv:
        cv_opts = literal_eval(alg_cv)
    else:
        cv_opts = {}
    y,y_t = train.train_predict(X,X_t,clf,cv_opts)
    logger.info('Train Error : {0}'.format(metrics.zero_one_loss(y,X['class'])))
    logger.info('Test Error : {0}'.format(metrics.zero_one_loss(y_t, X_t['class'])))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(obj={})
