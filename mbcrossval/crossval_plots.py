# -*- coding: utf-8 -*-
# Plot
import matplotlib
#matplotlib.use('TkAgg')  # noqa
import matplotlib.pyplot as plt

import os

# Libs
import numpy as np
import geopandas as gpd
import pandas as pd
import pickle

from oggm import utils, workflow, cfg


def crossval_boxwhisker(xval):

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


def crossval_timeseries(file, pltdir):
    # load pickle file
    xvaldict = pickle.load(open(file, 'rb'))
    data = xvaldict['massbalance']
    # time series plots of mass balance

    # reindex for plotting
    reind = pd.Index(np.arange(data.index[0], data.index[-1]+1))

    for gd in data.columns.levels[0]:
        f, ax1 = plt.subplots(1, 1, figsize=(12, 5), sharey=True)

        ax1.plot(data[gd].measured.reindex(reind), 'ko-', linewidth=3,
                 label='Measured annual mass balance',
                 color='xkcd:charcoal')
        ax1.plot(data[gd].calibrated.reindex(reind), 'go-', linewidth=3,
                 label='OGGM: Calibrated t_star',
                 color='xkcd:bluish')
        ax1.plot(data[gd].crossvalidated.reindex(reind), 'ro-', linewidth=3,
                 label='OGGM: Crossvalidated t_star',
                 color='xkcd:reddish')
        ax1.set_xlabel('Years')
        ax1.set_ylabel('Specific mass-balance (mm w.e.)')
        ax1.legend(loc='best')

        name = xvaldict['per_glacier'].loc[gd].Name

        if name == '':
            ax1.set_title(gd)
        else:
            ax1.set_title('%s (%s)' % (gd, name))

        ax1.grid(True)
        f.tight_layout()
        plotname = os.path.join(pltdir, '%s.png' % gd)
        f.savefig(plotname, format='png')
        plt.close(f)


def crossval_histogram(file, pltdir):
    # histogramplot of the crossvalidation. compare Marzeion 2012, Figure 3
    # load pickle file
    xvaldict = pickle.load(open(file, 'rb'))
    data = xvaldict['per_glacier']

    # Marzeion et al Figure 3
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    bins = np.arange(20) * 400 - 3800
    data['xval_bias'].plot(ax=ax1, kind='hist', bins=bins,
                            color='C3', label='')
    ax1.vlines(data['xval_bias'].mean(), 0, 120,
               linestyles='--', label='Mean')
    ax1.vlines(data['xval_bias'].quantile(), 0, 120, label='Median')
    ax1.vlines(data['xval_bias'].quantile([0.05, 0.95]), 0, 120,
               color='grey',
               label='5% and 95%\npercentiles')
    ax1.text(0.01, 0.99, 'N = {}'.format(len(data)),
             horizontalalignment='left',
             verticalalignment='top',
             transform=ax1.transAxes)

    ax1.set_ylim(0, 120)
    ax1.set_ylabel('N Glaciers')
    ax1.set_xlabel('Mass-balance error (mm w.e. yr$^{-1}$)')
    ax1.legend(loc='best')
    ax1.set_title('Cross validated t_star')
    data['interp_bias'].plot(ax=ax2, kind='hist', bins=bins, color='C0')
    ax2.vlines(data['interp_bias'].mean(), 0, 120, linestyles='--')
    ax2.vlines(data['interp_bias'].quantile(), 0, 120)
    ax2.vlines(data['interp_bias'].quantile([0.05, 0.95]), 0, 120,
               color='grey')
    ax2.set_xlabel('Mass-balance error (mm w.e. yr$^{-1}$)')
    ax2.set_title('Interpolated mu_star')
    plotname = os.path.join(pltdir, 'mb_histogram.png')
    plt.tight_layout()
    plt.savefig(plotname, format='png')
