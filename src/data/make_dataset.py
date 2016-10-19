# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import click
import logging
import src.settings as settings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


@click.command()
@click.option('--kfold',type=click.INT,default=10)
def main(kfold):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    raw_data_dir = os.path.join(settings.project_dir,os.environ.get('RAW_DATA_DIR'))
    processed_data_dir = os.path.join(settings.project_dir,os.environ.get('PROCESSED_DATA_DIR'))
    for dir_name in [x for x in os.listdir(raw_data_dir) if not x.startswith('.')]:
        logger.info('opening directory : {0}'.format(dir_name))
        current_dir = os.path.join(raw_data_dir, dir_name)
        data_files = os.listdir(current_dir)
        if 'data.csv' in data_files and os.path.isfile(os.path.join(current_dir,'data.csv')):
            logger.info('"data.csv" file has been found. Checking if it\'s correct')
            data_file_path = os.path.join(current_dir, 'data.csv')
            try:
                data = pd.read_csv(data_file_path)
                logger.info('File is loaded.Has shape : {0}'.format(data.shape))
            except Exception:
                logger.error('Error Reading the file. Moving on')
                continue
            try:
                os.mkdir(os.path.join(processed_data_dir,dir_name))
            except OSError:
                logger.info('Dir already exists.')
            pred = list(data.columns)
            pred.remove('class')
            sss = StratifiedShuffleSplit(n_splits=kfold,test_size=0.1)
            fold_index = 0
            for train_index, test_index in sss.split(data[pred].values, data['class'].values):
                np.save(os.path.join(processed_data_dir,dir_name,'train.{0}'.format(fold_index)),train_index)
                np.save(os.path.join(processed_data_dir, dir_name, 'test.{0}'.format(fold_index)), test_index)
                fold_index += 1
            data.to_csv(os.path.join(processed_data_dir,dir_name,'data.csv'),index=False)

        elif 'data.csv' not in data_files and 'data' in data_files and 'trueclass' in data_files:
            logger.info('"data.csv" not found, will create it.')
            data = pd.read_csv(os.path.join(current_dir, 'data'),header=None,delim_whitespace=True)
            data = data.loc[:,(data !=0).any()]
            trueclass = pd.read_csv(os.path.join(current_dir, 'trueclass'),header=None,delim_whitespace=True)
            pred = ['col_{0}'.format(x) for x in range(data.shape[1])]
            data.columns = pred
            trueclass = trueclass[0]
            try:
                os.mkdir(os.path.join(processed_data_dir,dir_name))
            except OSError:
                logger.info('Dir already exists.')

            sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.1)
            fold_index = 0
            for train_index, test_index in sss.split(data[pred].values, trueclass.values):
                np.save(os.path.join(processed_data_dir, dir_name, 'train.{0}'.format(fold_index)), train_index)
                np.save(os.path.join(processed_data_dir, dir_name, 'test.{0}'.format(fold_index)), test_index)
                fold_index += 1
            data = pd.concat([data,trueclass],axis=1)
            pred.append('class')
            data.columns = pred
            data.to_csv(os.path.join(processed_data_dir,dir_name,'data.csv'),index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
