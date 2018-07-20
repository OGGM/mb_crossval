# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa
import logging

import os
import datetime

# Libs
import numpy as np
import pandas as pd
import pickle
import argparse

# Local imports
from oggm import cfg, utils
from mbcrossval.crossvalidation import initialization_selection, preprocessing
from mbcrossval.crossvalidation import calibration, minor_xval_statistics
from mbcrossval.crossval_plots import crossval_timeseries, crossval_histogram
from mbcrossval.crossval_plots import crossval_boxplot
from mbcrossval.crossval_website import create_website
from mbcrossval import mbcfg

# Module logger
log = logging.getLogger(__name__)


def histalp():
    # If HISTALP run, only region 11 is valid:
    mbcfg.PARAMS['region'] = '11'
    url = 'https://cluster.klima.uni-bremen.de/~mdusch/' + \
          'histalp_merged_full_1850.nc.tar.gz'
    histalptar = utils.file_downloader(url)

    import tarfile
    with tarfile.open(histalptar, "r") as tar:
        tar.extractall(path=os.path.split(histalptar)[0])
        mbcfg.PATHS['histalpfile'] = os.path.join(os.path.split(histalptar)[0],
                                                  tar.getmembers()[0].name)
    if not os.path.exists(mbcfg.PATHS['histalpfile']):
        raise RuntimeError('Histalpfile not found')


def run_major_crossvalidation():
    # Initialize OGGM and set up the run parameters

    if mbcfg.PARAMS['histalp']:
        histalp()

    # initialization, select region if wanted and return GDIRs
    gdirs = initialization_selection()

    # some preprocessing
    gdirs = preprocessing(gdirs)

    # ----- cross validation loop

    # dataframe to store results
    xval = pd.DataFrame([], columns=['prcpsf', 'tliq', 'tmelt', 'tgrad',
                                     'std_quot', 'bias', 'rmse', 'core'])

    # define each parameter range
    prcpsf = np.arange(mbcfg.PARAMS['prcp1'],
                       mbcfg.PARAMS['prcp2'] + mbcfg.PARAMS['prcp_step'],
                       mbcfg.PARAMS['prcp_step'])
    tliq = np.arange(mbcfg.PARAMS['tliq1'],
                     mbcfg.PARAMS['tliq2'] + mbcfg.PARAMS['tliq_step'],
                     mbcfg.PARAMS['tliq_step'])
    tmelt = np.arange(mbcfg.PARAMS['tmel1'],
                      mbcfg.PARAMS['tmel2'] + mbcfg.PARAMS['tmel_step'],
                      mbcfg.PARAMS['tmel_step'])
    tgrad = np.arange(mbcfg.PARAMS['tgra1'],
                      mbcfg.PARAMS['tgra2'] + mbcfg.PARAMS['tgra_step'],
                      mbcfg.PARAMS['tgra_step'])

    # loop over all
    for PR in prcpsf:
        cfg.PARAMS['prcp_scaling_factor'] = PR
        for TL in tliq:
            cfg.PARAMS['temp_all_liq'] = TL
            for TM in tmelt:
                cfg.PARAMS['temp_melt'] = TM
                for TG in tgrad:
                    cfg.PARAMS['temp_default_gradient'] = TG

                    # execute calibraion and crossvalidation
                    log.info('prcpSF={}, Tliq={}, Tmelt={}, Tgrad={}'.format(
                        PR, TL, TM, TG))
                    xval = calibration(gdirs, xval, major=1)

    outdict = {'statistic': xval,
               'date_created': datetime.datetime.now().strftime('%Y-%m-%d'),
               'oggmversion': mbcfg.PARAMS['oggmversion']}

    pout = os.path.join(mbcfg.PATHS['storage_dir'],
                        'xval_%s_major.p' % mbcfg.PARAMS['oggmversion'])
    pickle.dump(outdict, open(pout, 'wb'))


def run_minor_crossvalidation():
    # Initialize OGGM and set up the run parameters

    if mbcfg.PARAMS['histalp']:
        histalp()

    # initialization, select region if wanted and return GDIRs
    gdirs = initialization_selection()

    # some preprocessing
    gdirs = preprocessing(gdirs)

    # ----- cross validation loop

    # dataframe to store results
    xval = pd.DataFrame([], columns=['prcpsf', 'tliq', 'tmelt', 'tgrad',
                                     'std_quot', 'bias', 'rmse', 'core'])

    xval = calibration(gdirs, xval, major=0)

    xval_perglacier, mb_perglacier = minor_xval_statistics(gdirs)

    outdict = {'statistic': xval,
               'per_glacier': xval_perglacier,
               'massbalance': mb_perglacier,
               'date_created': datetime.datetime.now().strftime('%Y-%m-%d'),
               'oggmversion': mbcfg.PARAMS['oggmversion']}

    pout = os.path.join(mbcfg.PATHS['storage_dir'],
                        'xval_%s_minor.p' % mbcfg.PARAMS['oggmversion'])
    pickle.dump(outdict, open(pout, 'wb'))


if __name__ == '__main__':

    # --------------------------------------------------------------
    # Arguments:
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='Paths/name of configuration file')

    args = parser.parse_args()

    # run configuration file
    mbcfg.initialize(args.config)

    # =============================================================
    # Run options
    #
  
    if mbcfg.PARAMS['run_minor_crossval']:
        run_minor_crossvalidation()

    if mbcfg.PARAMS['run_major_crossval']:
        run_major_crossvalidation()

    if mbcfg.PARAMS['make_minor_plots']:
        file = os.path.join(mbcfg.PATHS['storage_dir'],
                            'xval_%s_minor.p' % mbcfg.PARAMS['oggmversion'])
        crossval_timeseries(file)
        crossval_histogram(file)

    # if mbcfg.PARAMS['make_major_plots']:
    #    file = os.path.join(mbcfg.PATHS['storage_dir'],
    #                        'xval_%s_major.p' % mbcfg.PARAMS['oggmversion'])
    #    crossval_boxplot(file)

    if mbcfg.PARAMS['make_website']:
        create_website()
