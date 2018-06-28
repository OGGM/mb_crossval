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

# Local imports
import oggm
from oggm import cfg
from crossvalidation import initialization_selection, preprocessing
from crossvalidation import calibration, minor_xval_statistics

# Module logger
log = logging.getLogger(__name__)


def run_major_crossvalidation(working_dir, region=None, rgi_version='6'):
    # Initialize OGGM and set up the run parameters

    # initialization, select region if wanted and return GDIRs
    gdirs = initialization_selection(working_dir)

    # some preprocessing
    gdirs = preprocessing(gdirs)

    # ----- cross validation loop

    # dataframe to store results
    xval = pd.DataFrame([], columns=['prcpsf', 'tliq', 'tmelt', 'tgrad',
                                     'std_quot', 'bias', 'rmse', 'core'])

    # define each parameter range
    # could also be done from a config file
    prcpsf = np.arange(1.5, 3.75, 0.25)
    tliq = np.arange(1.0, 3.25, 0.25)
    tmelt = np.arange(-2.0, -0.25, 0.25)
    tgrad = np.arange(-0.0075, -0.00525, 0.001)

    totalruns = prcpsf.size * tliq.size * tmelt.size * tgrad.size
    runs = 0

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
                    xval = calibration(gdirs, xval)

                    if runs % np.round(totalruns/100) == 0:
                        log.info('%3d percent of total runs' %
                                 runs/np.round(totalruns/100))
                    runs += 1

    outdict = {'statistic': xval,
               'date_created': datetime.datetime.now().strftime('%Y-%m-%d'),
               'oggmversion': oggm.__version__}

    pout = os.path.join(working_dir, 'xval_major.p')
    pickle.dump(outdict, open(pout, 'wb'))


def run_minor_crossvalidation(working_dir, region=None, rgi_version='6'):
    # Initialize OGGM and set up the run parameters

    # initialization, select region if wanted and return GDIRs
    gdirs = initialization_selection(working_dir)

    # some preprocessing
    gdirs = preprocessing(gdirs)

    # ----- cross validation loop

    # dataframe to store results
    xval = pd.DataFrame([], columns=['prcpsf', 'tliq', 'tmelt', 'tgrad',
                                     'std_quot', 'bias', 'rmse', 'core'])

    xval = calibration(gdirs, xval)

    xval_perglacier, mb_perglacier = minor_xval_statistics(gdirs)

    outdict = {'statistic': xval,
               'per_glacier': xval_perglacier,
               'massbalance': mb_perglacier,
               'date_created': datetime.datetime.now().strftime('%Y-%m-%d'),
               'oggmversion': oggm.__version__}

    pout = os.path.join(working_dir, 'xval_minor.p')
    pickle.dump(outdict, open(pout, 'wb'))


if __name__ == '__main__':

    # --------------------------------------------------------------
    # Configuration parameters:
    #   Later either pass them as parameters or as config-File
    #
    # Limit crossvalidation to a single RGI region. Either for reduced runtime.
    #   Or due to regional climate data (e.g. HISTALP)
    #   Alps = '11', All reference glaciers = None
    region = '11'
    #
    # RGI Version
    rgi_version = '6'
    #
    # OGGM working directory
    working_dir = '/home/matthias/crossvalidate_oggm_parameters/tmp'
    #
    # Website root directory
    webroot = ''
    #
    # Decide which tasks to run:
    #
    # run_major_crossvalidataion runs the crossvalidation for a parameter range
    #   this is time consuming and should only be done for major OGGM updates
    run_major_crossval = 0
    #
    # run_minor_crossvalidataion runs the crossvalidation with the
    #   standard parameter. Only necessary if major_crossval is not run.
    run_minor_crossval = 1
    #
    # decide if crossvalidation plots are made or not
    #   If run_major_crossvalidataion == 1, the boxwhisker plots will be made
    make_crossval_plots = 1
    #
    # decide if a website will be created from the results
    website = 0
    #
    # --------------------------------------------------------------
    #

    # =============================================================
    # Run options
    #
    if run_major_crossval:
        run_major_crossvalidation(working_dir)

    if run_minor_crossval:
        run_minor_crossvalidation(working_dir, region=region)

    # if make_crossval_plots:
    #     make_crossvalidation_plots

    # if website:
    #     make_website(webroot)
