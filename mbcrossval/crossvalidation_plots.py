# -*- coding: utf-8 -*-
# Plot
import matplotlib
matplotlib.use('TkAgg')  # noqa
import matplotlib.pyplot as plt

import os

# Libs
import numpy as np
import geopandas as gpd
import pandas as pd

from oggm import utils, workflow, cfg


def crossvalidation_boxwhisker(xval):

    # for convenience
    xval.tgrad *= 1000
    xval.tgrad = np.round(xval.tgrad, decimals=1)

    # some plotting stuff:
    labels = {'prcpsf': 'Precipitation Factor',
              'tliq': 'T liquid precipitation [deg C]',
              'tmelt': 'Melt temperature [deg C]',
              'tgrad': 'Temperature laps rate [K/km]'}
    # some plotting stuff:
    title = {'prcpsf': 'Precipitation Factor',
             'tliq': 'Liquid precipitation temperature',
             'tmelt': 'Melt temperature',
             'tgrad': 'Temperature laps rate'}

    allvar = {'prcpsf': 2.5, 'tliq': 2.0, 'tmelt': -1.0, 'tgrad': -6.5}

    for var in allvar.keys():
        f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(13, 7))

        # find the entries with the standard values
        var0 = allvar.copy()
        del var0[var]
        idx = list(var0.keys())

        base = xval.loc[np.isclose(xval[idx[0]], var0[idx[0]]) &
                        np.isclose(xval[idx[1]], var0[idx[1]]) &
                        np.isclose(xval[idx[2]], var0[idx[2]])]

        # RMSE
        xval.boxplot(column='rmse', by=var, ax=ax0, grid=False,
                     positions=base[var], widths=0.2)
        base.plot(x=var, y='rmse', kind='scatter', ax=ax0, color='r',
                  linewidth=3, )
        ax0.set_ylabel('mean rmse')
        ax0.set_xlabel('')
        ax0.set_title('')
        ax0.set_ylim((200, 800))

        # BIAS
        xval.boxplot(column='bias', by=var, ax=ax1, grid=False,
                     positions=base[var], widths=0.2)
        base.plot(x=var, y='bias', kind='scatter', ax=ax1, color='r',
                  linewidth=3)
        ax1.set_ylabel('mean bias')
        ax1.set_xlabel('')
        ax1.set_title('')
        ax1.set_ylim((-400, 100))

        # STD quotient
        xval.boxplot(column='std_quot', by=var, ax=ax2, grid=False,
                     positions=base[var], widths=0.2)
        base.plot(x=var, y='std_quot', kind='scatter', ax=ax2, color='r',
                  linewidth=3)
        ax2.set_xlabel(labels[var])
        ax2.set_ylabel('mean std quotient')
        ax2.set_title('')
        ax2.set_ylim((0, 3))

        # CORE
        xval.boxplot(column='core', by=var, ax=ax3, grid=False,
                     positions=base[var], widths=0.2)
        base.plot(x=var, y='core', kind='scatter', ax=ax3, color='r',
                  linewidth=3)
        ax3.set_xlabel(labels[var])
        ax3.set_ylabel('mean corelation')
        ax3.set_title('')
        ax3.set_ylim((0.55, 0.65))

        # figure stuff

        f.suptitle('Crossvalidation results with respect to %s' % title[var])

        # f.tight_layout()
        f.savefig(var + '_crossval_box.png', format='png')

    plt.show()


def crossvalidation_timeseries(data):
    # time series plots of mass balance

    for gd in gdirs:
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


def crossvalidation_histogram(vdir):
    # histogramplot of the crossvalidation. compare Marzeion 2012, Figure 3

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
