from __future__ import print_function
from os.path import join
from os import environ
import click
import logging






def _transform(X,X_t,y,clf):
    clf.fit(X,y)
    return clf.predict(X),clf.predict(X_t)


