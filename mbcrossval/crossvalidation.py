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

# Local imports
from oggm import cfg, utils, tasks, workflow
from oggm.workflow import execute_entity_task
from oggm.core.massbalance import PastMassBalance

from mbcrossval import mbcfg

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

    # Climate tasks
    if mbcfg.PARAMS['histalp']:
        cfg.PATHS['climate_file'] = mbcfg.PATHS['histalpfile']
        execute_entity_task(tasks.process_custom_climate_data, gdirs)
    else:
        execute_entity_task(tasks.process_cru_data, gdirs)

    return gdirs


def calibration(gdirs, xval, major=0):

    # once for reference t_stars
    with utils.DisableLogger():
        tasks.compute_ref_t_stars(gdirs)
        execute_entity_task(tasks.local_mustar, gdirs)
        execute_entity_task(tasks.apparent_mb, gdirs)

    # do the crossvalidation
    xval = quick_crossval(gdirs, xval, major=major)

    return xval


def quick_crossval(gdirs, xval, major=0):
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
            tasks.local_mustar(gdir, ref_df=tmp_ref_df)

        # read crossvalidated values
        rdf = pd.read_csv(gdir.get_filepath('local_mustar'))

        # ----
        # --- MASS-BALANCE MODEL
        heights, widths = gdir.get_inversion_flowline_hw()
        mb_mod = PastMassBalance(gdir,
                                 mu_star=rdf['mu_star'].values[0],
                                 bias=rdf['bias'].values[0],
                                 )

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

        if not major:
            # store cross validated values
            full_ref_df.loc[rid, 'cv_tstar'] = int(rdf['t_star'].values[0])
            full_ref_df.loc[rid, 'cv_mustar'] = rdf['mu_star'].values[0]
            full_ref_df.loc[rid, 'cv_bias'] = rdf['bias'].values[0]
            full_ref_df.loc[rid, 'cv_prcp_fac'] =\
                cfg.PARAMS['prcp_scaling_factor']

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

    if major:
        return xval
    else:
        for i, rid in enumerate(full_ref_df.index):
            # the glacier to look at
            gdir = full_ref_df.loc[full_ref_df.index == rid]
            # the reference glaciers
            tmp_ref_df = full_ref_df.loc[full_ref_df.index != rid]

            # Compute the distance
            distances = utils.haversine(gdir.lon.values[0], gdir.lat.values[0],
                                        tmp_ref_df.lon, tmp_ref_df.lat)

            # Take the 10 closests
            aso = np.argsort(distances)[0:9]
            amin = tmp_ref_df.iloc[aso]
            distances = distances[aso] ** 2
            interp = np.average(amin.mustar, weights=1. / distances)
            full_ref_df.loc[rid, 'interp_mustar'] = interp
        # write
        file = os.path.join(cfg.PATHS['working_dir'], 'crossval_tstars.csv')
        full_ref_df.to_csv(file)
        # alternative: do not write csv file, but store the needed values
        # within xval_minor_statistics

        return xval


def initialization_selection():
    # -------------
    # Initialization
    # -------------
    cfg.initialize()

    # working directories
    cfg.PATHS['working_dir'] = mbcfg.PATHS['working_dir']

    cfg.PATHS['rgi_version'] = mbcfg.PARAMS['rgi_version']

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
        if mbcfg.PARAMS['region'] is not None\
                and reg != mbcfg.PARAMS['region']:
            continue

        fn = '*' + reg + '_rgi{}0_*.shp'.format(cfg.PATHS['rgi_version'])
        fs = list(sorted(glob(os.path.join(rgi_dir, '*', fn))))[0]
        sh = gpd.read_file(fs)
        rgidf.append(sh.loc[sh.RGIId.isin(rids)])
    rgidf = pd.concat(rgidf)
    rgidf.crs = sh.crs  # for geolocalisation

    # reduce Europe to Histalp area (exclude Pyrenees, etc...)
    if mbcfg.PARAMS['histalp']:
        rgidf = rgidf.loc[(rgidf.CenLon >= 4) &
                          (rgidf.CenLon < 20) &
                          (rgidf.CenLat >= 43) &
                          (rgidf.CenLat < 47)]

        # and set standard histalp values
        cfg.PARAMS['prcp_scaling_factor'] = 1.75
        cfg.PARAMS['temp_all_liq'] = 2.0
        cfg.PARAMS['temp_melt'] = -1.75
        cfg.PARAMS['temp_default_gradient'] = -0.0065


    # We have to check which of them actually have enough mb data.
    # Let OGGM do it:
    gdirs = workflow.init_glacier_regions(rgidf)
    # We need to know which period we have data for

    if mbcfg.PARAMS['histalp']:
        cfg.PATHS['climate_file'] = mbcfg.PATHS['histalpfile']
        execute_entity_task(tasks.process_custom_climate_data, gdirs)
    else:
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
    gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)

    return gdirs


def minor_xval_statistics(gdirs):
    # initialize the pandas dataframes

    # to store mass balances of every glacier
    mbdf = pd.DataFrame([], index=np.arange(1850, 2050))

    # Cross-validation
    file = os.path.join(cfg.PATHS['working_dir'], 'crossval_tstars.csv')
    cvdf = pd.read_csv(file, index_col=0)

    # dataframe output
    xval = pd.DataFrame([], columns=['RGIId',
                                     'Name',
                                     'tstar_bias',
                                     'xval_bias',
                                     'interp_bias',
                                     'mustar',
                                     'tstar',
                                     'xval_mustar',
                                     'xval_tstar',
                                     'interp_mustar'])

    for gd in gdirs:
        t_cvdf = cvdf.loc[gd.rgi_id]
        heights, widths = gd.get_inversion_flowline_hw()

        # Observed mass-blance
        refmb = gd.get_ref_mb_data().copy()

        # Mass-balance model with cross-validated parameters instead
        mb_mod = PastMassBalance(gd, mu_star=t_cvdf.cv_mustar,
                                 bias=t_cvdf.cv_bias,
                                 )
        refmb['OGGM_cv'] = mb_mod.get_specific_mb(heights, widths,
                                                  year=refmb.index)
        # Compare their standard deviation
        std_ref = refmb.ANNUAL_BALANCE.std()
        rcor = np.corrcoef(refmb.OGGM_cv, refmb.ANNUAL_BALANCE)[0, 1]
        if std_ref == 0:
            # I think that such a thing happens with some geodetic values
            std_ref = refmb.OGGM_cv.std()
            rcor = 1
        # Store the scores
        cvdf.loc[gd.rgi_id, 'CV_MB_BIAS'] = (refmb.OGGM_cv.mean() -
                                             refmb.ANNUAL_BALANCE.mean())
        cvdf.loc[gd.rgi_id, 'CV_MB_SIGMA_BIAS'] = (refmb.OGGM_cv.std() /
                                                   std_ref)
        cvdf.loc[gd.rgi_id, 'CV_MB_COR'] = rcor

        # Mass-balance model with interpolated mu_star
        mb_mod = PastMassBalance(gd, mu_star=t_cvdf.interp_mustar,
                                 bias=t_cvdf.cv_bias,
                                 )
        refmb['OGGM_mu_interp'] = mb_mod.get_specific_mb(heights, widths,
                                                         year=refmb.index)
        cvdf.loc[gd.rgi_id, 'INTERP_MB_BIAS'] = (refmb.OGGM_mu_interp.mean() -
                                                 refmb.ANNUAL_BALANCE.mean())

        # Mass-balance model with best guess tstar
        mb_mod = PastMassBalance(gd, mu_star=t_cvdf.mustar,
                                 bias=t_cvdf.bias,
                                 )
        refmb['OGGM_tstar'] = mb_mod.get_specific_mb(heights, widths,
                                                     year=refmb.index)
        cvdf.loc[gd.rgi_id, 'tstar_MB_BIAS'] = (refmb.OGGM_tstar.mean() -
                                                refmb.ANNUAL_BALANCE.mean())

        # Pandas DataFrame Output
        #
        # 1. statistics
        tbias = cvdf.loc[gd.rgi_id, 'tstar_MB_BIAS']
        xbias = cvdf.loc[gd.rgi_id, 'CV_MB_BIAS']
        ibias = cvdf.loc[gd.rgi_id, 'INTERP_MB_BIAS']
        xval = xval.append({'Name': gd.name,
                            'RGIId': gd.rgi_id,
                            'tstar_bias': tbias,
                            'xval_bias': xbias,
                            'interp_bias': ibias,
                            'mustar': t_cvdf.mustar,
                            'tstar': t_cvdf.tstar,
                            'xval_mustar': t_cvdf.cv_mustar,
                            'xval_tstar': t_cvdf.cv_tstar,
                            'interp_mustar': t_cvdf.interp_mustar},
                           ignore_index=True)

        #
        # 2. mass balance timeseries
        mbarray = np.dstack((refmb.ANNUAL_BALANCE,
                             refmb.OGGM_tstar,
                             refmb.OGGM_cv)).squeeze()

        mbdf_add = pd.DataFrame(mbarray,
                                columns=[[gd.rgi_id, gd.rgi_id, gd.rgi_id],
                                         ['measured',
                                          'calibrated',
                                          'crossvalidated']],
                                index=refmb.index)
        mbdf = pd.concat([mbdf, mbdf_add], axis=1)

    mbdf.columns = pd.MultiIndex.from_tuples(mbdf.columns)

    mbdf = mbdf.dropna(how='all')

    xval.index = xval.RGIId

    return xval, mbdf
