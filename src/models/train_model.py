import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV


def train_predict(X,X_t,clf,cv_opts):
    logger = logging.getLogger(__name__)
    grid_search = GridSearchCV(clf, cv_opts, n_jobs=-1,verbose=1)
    pred = [x for x in X.columns if x != 'class']
    logger.info('Do Grid Search')
    grid_search.fit(X[pred],X['class'])
    best_clf = grid_search.best_estimator_
    return best_clf.predict(X[pred]),best_clf.predict(X_t[pred])