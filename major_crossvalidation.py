# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa
import logging

import os
from glob import glob

# Libs
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle

# Local imports
import oggm
from oggm import cfg, utils, tasks, workflow
from oggm.workflow import execute_entity_task
from oggm.core.massbalance import PastMassBalance

# Module logger
log = logging.getLogger(__name__)

def preprocessing(gdirs):
    # Prepro tasks
    task_list = [
        tasks.glacier_masks,
        tasks.compute_centerlines,
        tasks.initialize_flowlines,
        tasks.catchment_area,
        tasks.catchment_intersections,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
    ]
    for task in task_list:
        execute_entity_task(task, gdirs)

    return gdirs


def calibration(gdirs, xval):
    # Climate tasks
    # do once per parameter set
    with utils.DisableLogger():
        execute_entity_task(tasks.process_cru_data, gdirs)
        tasks.compute_ref_t_stars(gdirs)
        tasks.distribute_t_stars(gdirs)
        execute_entity_task(tasks.apparent_mb, gdirs)

    # do the crossvalidation
    xval = quick_crossval(gdirs, xval)

    return xval


def quick_crossval(gdirs, xval):
    # following climate.quick_crossval_t_stars
    # but minimized for performance

    full_ref_df = pd.read_csv(os.path.join(cfg.PATHS['working_dir'],
                                           'ref_tstars.csv'), index_col=0)

    tmpdf = pd.DataFrame([], columns=['std_oggm',
                                      'std_ref',
                                      'rmse',
                                      'core',
                                      'bias'])

    for i, rid in enumerate(full_ref_df.index):

        # the glacier to look at
        gdir = [g for g in gdirs if g.rgi_id == rid][0]

        # the reference glaciers
        tmp_ref_df = full_ref_df.loc[full_ref_df.index != rid]

        # select reference glacier directories
        # Only necessary if tasks.compute_ref_t_stars is uncommented below
        # ref_gdirs = [g for g in gdirs if g.rgi_id != rid]

        # before the cross-val store the info about "real" mustar
        rdf = pd.read_csv(gdir.get_filepath('local_mustar'))
        full_ref_df.loc[rid, 'mustar'] = rdf['mu_star'].values[0]

        # redistribute t_star
        with utils.DisableLogger():
            # compute_ref_t_stars should be done again for
            # every crossvalidation step
            # This will/might have an influence if one of the 10 surrounding
            # glaciers of the current glacier has more than one t_star
            # If so, the currently crossvalidated glacier was probably
            # used to select one t_star for this surrounding glacier.
            #
            # But: compute_ref_t_stars is very time consuming. And the
            # influence is probably very small. Also only 40 out of the 253
            # reference glaciers do have more than one possible t_star.
            #
            # tasks.compute_ref_t_stars(ref_gdirs)
            tasks.distribute_t_stars([gdir], ref_df=tmp_ref_df)

        # read crossvalidated values
        rdf = pd.read_csv(gdir.get_filepath('local_mustar'))

        # ----
        # --- MASS-BALANCE MODEL
        heights, widths = gdir.get_inversion_flowline_hw()
        mb_mod = PastMassBalance(gdir,
                                 mu_star=rdf['mu_star'].values[0],
                                 bias=rdf['bias'].values[0],
                                 prcp_fac=rdf['prcp_fac'].values[0])

        # Mass-blaance timeseries, observed and simulated
        refmb = gdir.get_ref_mb_data().copy()
        refmb['OGGM'] = mb_mod.get_specific_mb(heights, widths,
                                               year=refmb.index)

        # store single glacier results
        bias = refmb.OGGM.mean() - refmb.ANNUAL_BALANCE.mean()
        rmse = np.sqrt(np.mean(refmb.OGGM - refmb.ANNUAL_BALANCE)**2)
        rcor = np.corrcoef(refmb.OGGM, refmb.ANNUAL_BALANCE)[0, 1]

        ref_std = refmb.ANNUAL_BALANCE.std()

        # unclear how to treat this best
        if ref_std == 0:
            ref_std = refmb.OGGM.std()
            rcor = 1

        tmpdf.loc[len(tmpdf.index)] = {'std_oggm': refmb.OGGM.std(),
                                       'std_ref': ref_std,
                                       'bias': bias,
                                       'rmse': rmse,
                                       'core': rcor
                                       }

    # and store mean values
    std_quot = np.mean(tmpdf.std_oggm/tmpdf.std_ref)

    xval.loc[len(xval.index)] = {'prcpsf': cfg.PARAMS['prcp_scaling_factor'],
                                 'tliq': cfg.PARAMS['temp_all_liq'],
                                 'tmelt': cfg.PARAMS['temp_melt'],
                                 'tgrad': cfg.PARAMS['temp_default_gradient'],
                                 'std_quot': std_quot,
                                 'bias': tmpdf['bias'].mean(),
                                 'rmse': tmpdf['rmse'].mean(),
                                 'core': tmpdf['core'].mean()}

    # maybe: do test here if MB is approximately 0 for each glacier!

    return xval


def initialization_selection(region=None):
    # -------------
    # Initialization
    # -------------

    # RGI Version
    rgi_version = '6'

    # Initialize OGGM and set up the run parameters
    cfg.initialize()

    # Local paths (where to write the OGGM run output)
    working_dir = os.path.join(os.path.expanduser('~'),
                               'crossvalidate_oggm_parameters',
                               'OGGM_RGIV{}_OGGM{}'.format(
                                   rgi_version, oggm.__version__))
    utils.mkdir(working_dir)
    cfg.PATHS['working_dir'] = working_dir
    cfg.PATHS['rgi_version'] = rgi_version

    # We are running the calibration ourselves
    cfg.PARAMS['run_mb_calibration'] = True

    # No need for intersects since this has an effect on the inversion only
    cfg.PARAMS['use_intersects'] = False

    # Use multiprocessing?
    cfg.PARAMS['use_multiprocessing'] = True

    # Set to True for operational runs
    # maybe also here?
    cfg.PARAMS['continue_on_error'] = False

    # set negative flux filtering to false. should be standard soon
    cfg.PARAMS['filter_for_neg_flux'] = False

    # Pre-download other files which will be needed later
    _ = utils.get_cru_file(var='tmp')
    _ = utils.get_cru_file(var='pre')
    rgi_dir = utils.get_rgi_dir(version=cfg.PATHS['rgi_version'])

    # Get the reference glacier ids (they are different for each RGI version)
    df, _ = utils.get_wgms_files()
    rids = df['RGI{}0_ID'.format(cfg.PATHS['rgi_version'])]

    # Make a new dataframe with those (this takes a while)
    rgidf = []
    for reg in df['RGI_REG'].unique():
        if reg == '19':
            continue  # we have no climate data in Antarctica
        if region is not None and reg != region:
            continue

        fn = '*' + reg + '_rgi{}0_*.shp'.format(cfg.PATHS['rgi_version'])
        fs = list(sorted(glob(os.path.join(rgi_dir, '*', fn))))[0]
        sh = gpd.read_file(fs)
        rgidf.append(sh.loc[sh.RGIId.isin(rids)])
    rgidf = pd.concat(rgidf)
    rgidf.crs = sh.crs  # for geolocalisation

    # We have to check which of them actually have enough mb data.
    # Let OGGM do it:
    gdirs = workflow.init_glacier_regions(rgidf)
    # We need to know which period we have data for
    execute_entity_task(tasks.process_cru_data, gdirs, print_log=False)
    gdirs = utils.get_ref_mb_glaciers(gdirs)
    # Keep only these
    rgidf = rgidf.loc[rgidf.RGIId.isin([g.rgi_id for g in gdirs])]

    # Save
    rgidf.to_file(os.path.join(cfg.PATHS['working_dir'],
                               'mb_ref_glaciers.shp'))

    # Sort for more efficient parallel computing
    rgidf = rgidf.sort_values('Area', ascending=False)

    # Go - initialize working directories
    gdirs = workflow.init_glacier_regions(rgidf)

    return gdirs


if __name__ == '__main__':

    # initialization, select region if wanted and return GDIRs
    # pass e.g. >> region='11' << to select the Alps only for computation time
    gdirs = initialization_selection()

    # some preprocessing
    gdirs = preprocessing(gdirs)

    # ----- cross validation loop

    # dataframe to store results
    xval = pd.DataFrame([], columns=['prcpsf', 'tliq', 'tmelt', 'tgrad',
                                     'std_quot', 'bias', 'rmse', 'core'])

    # define each parameter range
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

                    if runs%np.round(totalruns/100) == 0:
                        log.info('%3d percent of total runs' %
                                 runs/np.round(totalruns/100))
                    runs += 1

    pickle.dump(xval, open('xval.p', 'wb'))
