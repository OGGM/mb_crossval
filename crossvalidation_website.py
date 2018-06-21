# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa

import os
from glob import glob
import shutil

# Plot
import matplotlib
matplotlib.use('TkAgg')  # noqa
import matplotlib.pyplot as plt

# Libs
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime

# Local imports
import oggm
from oggm import cfg, utils, tasks, workflow
from oggm.workflow import execute_entity_task
from oggm.core.massbalance import PastMassBalance


def calibration_crossvalidation():
    # MB crossvalidation with standard parameters

    # We are running the calibration ourselves
    cfg.PARAMS['run_mb_calibration'] = True

    # No need for intersects since this has an effect on the inversion only
    cfg.PARAMS['use_intersects'] = False

    # Use multiprocessing?
    cfg.PARAMS['use_multiprocessing'] = True

    # Set to True for operational runs
    # maybe also here?
    cfg.PARAMS['continue_on_error'] = False

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
    execute_entity_task(tasks.process_cru_data, gdirs)
    tasks.compute_ref_t_stars(gdirs)
    tasks.distribute_t_stars(gdirs)
    execute_entity_task(tasks.apparent_mb, gdirs)
    # Recompute after the first round - this is being picky but this is
    # Because geometries may change after apparent_mb's filtering
    tasks.compute_ref_t_stars(gdirs)
    tasks.distribute_t_stars(gdirs)
    execute_entity_task(tasks.apparent_mb, gdirs)

    # Model validation
    tasks.quick_crossval_t_stars(gdirs)  # for later
    tasks.distribute_t_stars(gdirs)  # To restore after cross-val

    # Tests: for all glaciers, the mass-balance around tstar and the
    # bias with observation should be approx 0
    from oggm.core.massbalance import (ConstantMassBalance, PastMassBalance)
    for gd in gdirs:
        heights, widths = gd.get_inversion_flowline_hw()

        mb_mod = ConstantMassBalance(gd, bias=0)  # bias=0 because of calib!
        mb = mb_mod.get_specific_mb(heights, widths)
        np.testing.assert_allclose(mb, 0, atol=10)  # numerical errors

        mb_mod = PastMassBalance(gd)  # Here we need the computed bias
        refmb = gd.get_ref_mb_data().copy()
        refmb['OGGM'] = mb_mod.get_specific_mb(heights, widths,
                                               year=refmb.index)
        np.testing.assert_allclose(refmb.OGGM.mean(),
                                   refmb.ANNUAL_BALANCE.mean(),
                                   atol=10)


def plt_crossvalidation(vdir):

    # plot directory
    pltdir = os.path.join(vdir, 'plots')
    utils.mkdir(pltdir)

    # Read the rgi file
    rgidf = gpd.read_file(os.path.join(cfg.PATHS['working_dir'],
                                       'mb_ref_glaciers.shp'))

    # Go - initialize working directories
    gdirs = workflow.init_glacier_regions(rgidf)

    # Cross-validation
    file = os.path.join(cfg.PATHS['working_dir'], 'crossval_tstars.csv')
    cvdf = pd.read_csv(file, index_col=0)

    # dataframe output
    df = pd.DataFrame([], columns=['RGIId',
                                   'Name',
                                   'Tstar bias',
                                   'Xval bias'])

    for gd in gdirs:
        t_cvdf = cvdf.loc[gd.rgi_id]
        heights, widths = gd.get_inversion_flowline_hw()

        # Observed mass-blance
        refmb = gd.get_ref_mb_data().copy()

        # Mass-balance model with cross-validated parameters instead
        mb_mod = PastMassBalance(gd, mu_star=t_cvdf.cv_mustar,
                                 bias=t_cvdf.cv_bias,
                                 prcp_fac=t_cvdf.cv_prcp_fac)
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
                                 prcp_fac=t_cvdf.cv_prcp_fac)
        refmb['OGGM_mu_interp'] = mb_mod.get_specific_mb(heights, widths,
                                                         year=refmb.index)
        cvdf.loc[gd.rgi_id, 'INTERP_MB_BIAS'] = (refmb.OGGM_mu_interp.mean() -
                                                 refmb.ANNUAL_BALANCE.mean())

        # Mass-balance model with best guess tstar
        mb_mod = PastMassBalance(gd, mu_star=t_cvdf.mustar,
                                 bias=t_cvdf.bias,
                                 prcp_fac=t_cvdf.prcp_fac)
        refmb['OGGM_tstar'] = mb_mod.get_specific_mb(heights, widths,
                                                     year=refmb.index)
        cvdf.loc[gd.rgi_id, 'tstar_MB_BIAS'] = (refmb.OGGM_tstar.mean() -
                                                refmb.ANNUAL_BALANCE.mean())

        f, ax1 = plt.subplots(1, 1, figsize=(12, 5), sharey=True)

        # reindex for plotting
        reind = pd.Index(np.arange(refmb.index[0],refmb.index[-1]+1))
        ax1.plot(refmb.ANNUAL_BALANCE.reindex(reind), 'ko-', linewidth=3,
                 label='Measured annual mass balance',
                 color='xkcd:charcoal')
        ax1.plot(refmb.OGGM_tstar.reindex(reind), 'go-', linewidth=3,
                 label='OGGM: Calibrated t_star',
                 color='xkcd:bluish')
        ax1.plot(refmb.OGGM_cv.reindex(reind), 'ro-', linewidth=3,
                 label='OGGM: Crossvalidated t_star',
                 color='xkcd:reddish')
        # ax1.set_ylim(-4000, 4000)
        ax1.set_ylabel('Years')
        ax1.set_ylabel('Specific mass-balance (mm w.e.)')
        ax1.legend(loc='best')

        if gd.name == '':
            ax1.set_title(gd.rgi_id)
        else:
            ax1.set_title(gd.rgi_id + ' (' + gd.name + ')')

        tbias = cvdf.loc[gd.rgi_id, 'tstar_MB_BIAS']
        xbias = cvdf.loc[gd.rgi_id, 'CV_MB_BIAS']
        df = df.append({'Name': gd.name,
                        'RGIId': gd.rgi_id,
                        'Tstar bias': tbias,
                        'Xval bias': xbias},
                       ignore_index=True)
        ax1.grid(True)
        f.tight_layout()
        plotname = os.path.join(pltdir, '%s.png' % (gd.rgi_id))
        f.savefig(plotname, format='png')
        plt.close(f)

    # Marzeion et al Figure 3
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    bins = np.arange(20) * 400 - 3800
    cvdf['CV_MB_BIAS'].plot(ax=ax1, kind='hist', bins=bins,
                            color='C3', label='')
    ax1.vlines(cvdf['CV_MB_BIAS'].mean(), 0, 120,
               linestyles='--', label='Mean')
    ax1.vlines(cvdf['CV_MB_BIAS'].quantile(), 0, 120, label='Median')
    ax1.vlines(cvdf['CV_MB_BIAS'].quantile([0.05, 0.95]), 0, 120,
               color='grey',
               label='5% and 95%\npercentiles')
    ax1.text(0.01, 0.99, 'N = {}'.format(len(gdirs)),
             horizontalalignment='left',
             verticalalignment='top',
             transform=ax1.transAxes)

    ax1.set_ylim(0, 120)
    ax1.set_ylabel('N Glaciers')
    ax1.set_xlabel('Mass-balance error (mm w.e. yr$^{-1}$)')
    ax1.legend(loc='best')
    ax1.set_title('Cross validated t_star')
    cvdf['INTERP_MB_BIAS'].plot(ax=ax2, kind='hist', bins=bins, color='C0')
    ax2.vlines(cvdf['INTERP_MB_BIAS'].mean(), 0, 120, linestyles='--')
    ax2.vlines(cvdf['INTERP_MB_BIAS'].quantile(), 0, 120)
    ax2.vlines(cvdf['INTERP_MB_BIAS'].quantile([0.05, 0.95]), 0, 120,
               color='grey')
    ax2.set_xlabel('Mass-balance error (mm w.e. yr$^{-1}$)')
    ax2.set_title('Interpolated mu_star')
    plotname = os.path.join(pltdir, 'mb_histogram.png')
    plt.tight_layout()
    plt.savefig(plotname, format='png')

    name = 'Overview'

    tbias = cvdf['tstar_MB_BIAS'].mean()
    xbias = cvdf['CV_MB_BIAS'].mean()

    # sort whole dataframe
    df.sort_values('Name', inplace=True)
    # move glaciers without name to the end

    # concatenate the overview to the beginning
    df = pd.concat([pd.DataFrame([{'Name': name,
                                   'RGIId' : np.nan,
                                   'Xval bias': xbias,
                                   'Tstar bias': tbias}]),
                    df],
                   ignore_index=True)

    df = pd.concat([df.loc[df.Name != ''], df.loc[df.Name == '']])

    # some other stuff

    df.loc[df.Name==name, 'date_created'] = datetime.\
        datetime.now().strftime('%Y-%m-%d')
    df.loc[df.Name==name, 'oggmversion'] = oggm.__version__
    outfile = os.path.join(vdir, 'data.csv')
    df.to_csv(outfile, index=False)


def make_website(jinjadir, webroot):
    from jinja2 import Environment, FileSystemLoader

    # setup jinja
    file_loader = FileSystemLoader(jinjadir)
    env = Environment(loader=file_loader)

    # directories of different versions
    vdirs = []

    for x in os.listdir(webroot):
        if (x[0] == 'v') & (x.replace('.', '').strip('v').isnumeric()):
            # web site dir
            vdirs.append(x)
            # webdir
            webdir = os.path.join(webroot, x, 'web')
            shutil.rmtree(webdir)
            utils.mkdir(webdir)

    vdirs.sort()

    for nr, vdir in enumerate(vdirs):

        # webdir
        webdir = os.path.join(webroot, vdir, 'web')

        # read data
        df = pd.read_csv(os.path.join(webroot, vdir, 'data.csv'))

        #
        # LINKNAMEs
        df['linkname'] = df.RGIId
        df.loc[~pd.isna(df.Name), 'linkname'] = df.loc[~pd.isna(df.Name),
                                                       'linkname'] + \
                                                ', ' + \
                                                df.loc[~pd.isna(df.Name),
                                                       'Name']
        df.loc[pd.isna(df.RGIId), 'linkname'] = 'Overview'

        #
        # LINKLIST for GLACIERS
        df['link'] = df.RGIId + '.html'
        df.loc[pd.isna(df.RGIId), 'link'] = '../index.html'
        template = env.get_template('createlinklist.txt')
        linklist = template.render(glaciers=df.to_dict(orient='records'))
        with open(os.path.join(jinjadir, 'linklist.html'), 'w') as fl:
            fl.write(linklist)

        #
        # LINKLIST for INDEX
        df['link'] = 'web/' + df.RGIId + '.html'
        df.loc[pd.isna(df.RGIId), 'link'] = 'index.html'
        linklist = template.render(glaciers=df.to_dict(orient='records'))
        with open(os.path.join(jinjadir, 'linklistindex.html'), 'w') as fl:
            fl.write(linklist)

        #
        # WRITE ACTUAL HTML FILES
        template = env.get_template('template.html')
        fallback = env.get_template('fallback.html')
        date = df.date_created.dropna().values[0]
        oggmversion = df.oggmversion.dropna().values[0]
        for idx, glc in df.iterrows():

            # DIFFERENT VALUES DEPENDING ON INDEX OR GLACIER
            if pd.isna(glc.RGIId):
                # first: index
                bias1 = 'Mean t_star bias:'.ljust(27) + \
                        "{0:5.1f}".format(glc['Tstar bias'])
                bias2 = 'Mean crossval bias:'.ljust(27) + \
                        "{0:5.1f}".format(glc['Xval bias'])

                htmlname = os.path.join(webroot, vdir, 'index.html')
                imgname = 'plots/mb_histogram.png'
                index = 1
                linksuffix = '../'

                # path to where the crossval plots SHOULD BE stored
                cvplots = [x for x in os.listdir(
                    os.path.join(webroot, vdir, 'plots')) if 'crossval' in x]
                if len(cvplots) > 0:
                    crossval = 1

            else:
                # second glaciers
                bias1 = '    Calibrated MB bias:'.ljust(35) + \
                        "{0:5.6f}".format(glc['Tstar bias'])
                bias2 = 'Crossvalidated MB bias:'.ljust(35) + \
                        "{0:5.1f}".format(glc['Xval bias'])

                htmlname = os.path.join(webdir, glc['RGIId']) + '.html'
                imgname = '../plots/%s.png' % glc['RGIId']
                index = 0
                linksuffix = '../../'
                crossval = 0
                cvplots = ''


            #
            # Add PREVIOUS/NEXT buttons and link them
            if (vdir == vdirs[0]) & (len(vdirs) > 1):
                # first version, no previous
                previous = ''
                nxtlink = os.path.join(linksuffix, vdirs[nr+1], glc['link'])
                nxtfile = os.path.join(webroot, vdirs[nr+1], glc['link'])

                next = '<a href="%s" class="next">%s &raquo;</a>' %\
                       (nxtlink, vdirs[nr+1])
                if not os.path.isfile(nxtfile):
                    fbhtml = fallback.render()
                    with open(nxtfile, 'w') as fb:
                        fb.write(fbhtml)

            elif (vdir == vdirs[-1]) & (len(vdirs) > 1):
                # last version, no next
                next = ''
                prvlink = os.path.join(linksuffix, vdirs[nr-1], glc['link'])
                prvfile = os.path.join(webroot, vdirs[nr-1], glc['link'])

                previous = '<a href="%s" class="previous">&laquo; %s</a>' % \
                           (prvlink, vdirs[nr-1])
                if not os.path.isfile(prvfile):
                    fbhtml = fallback.render()
                    with open(prvfile, 'w') as fb:
                        fb.write(fbhtml)

            elif len(vdirs) == 1:
                next = ''
                previous = ''
            else:
                nxtlink = os.path.join(linksuffix, vdirs[nr+1], glc['link'])
                nxtfile = os.path.join(webroot, vdirs[nr+1], glc['link'])
                prvlink = os.path.join(linksuffix, vdirs[nr-1], glc['link'])
                prvfile = os.path.join(webroot, vdirs[nr-1], glc['link'])

                previous = '<a href="%s" class="previous">&laquo; %s</a>' % \
                           (prvlink, vdirs[nr-1])
                next = '<a href="%s" class="next">%s &raquo;</a>' % \
                       (nxtlink, vdirs[nr+1])
                if not os.path.isfile(nxtfile):
                    fbhtml = fallback.render()
                    with open(nxtfile, 'w') as fb:
                        fb.write(fbhtml)
                if not os.path.isfile(prvfile):
                    fbhtml = fallback.render()
                    with open(prvfile, 'w') as fb:
                        fb.write(fbhtml)

            glchtml = template.render(glcname=glc['linkname'],
                                      glcimg=imgname,
                                      version=oggmversion,
                                      date=date,
                                      bias1=bias1,
                                      bias2=bias2,
                                      index=index,
                                      cvplots=cvplots,
                                      crossval=crossval,
                                      previous=previous,
                                      next=next)
            with open(htmlname, 'w') as fl:
                fl.write(glchtml)

        # for the latest version: Create a index.html in the webroot
        if vdir == vdirs[-1]:
            template = env.get_template('latestindex.html')
            latest = template.render(version=vdir)
            latestfile = os.path.join(webroot, 'index.html')

            with open(latestfile, 'w') as fl:
                fl.write(latest)



if __name__ == '__main__':
    # -------------
    # Initialization
    # -------------

    # Initialize OGGM and set up the run parameters
    cfg.initialize()

    # RGI Version
    rgi_version = '6'
    cfg.PATHS['rgi_version'] = rgi_version

    # Local paths (where to write the OGGM run output and web page)
    workingroot = '.'
    # webroot is most likely different than the workingroot
    # e.g. cluster /home/www/oggm
    webroot = '.'

    # working directory
    working_dir = os.path.join(workingroot, 'tmp')
    cfg.PATHS['working_dir'] = working_dir
    utils.mkdir(working_dir)

    # subdir for current version
    version = 'v1.0.0'
    vdir = os.path.join(webroot, version)
    utils.mkdir(vdir)

    # directory where jinja is stored
    jinjadir = os.path.join(workingroot, 'jinja')

    # make new run or just redo the website?
    newrun = 1

    if newrun:
        calibration_crossvalidation()
        plt_crossvalidation(vdir)

    # every folder with the form 'v1.0.0' will be turned into a website
    make_website(jinjadir, webroot)

    # finally remove temporary working directory
    if newrun:
        shutil.rmtree(working_dir)
