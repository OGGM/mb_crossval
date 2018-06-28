# -*- coding: utf-8 -*-

# Plot
import matplotlib
matplotlib.use('TkAgg')  # noqa
import matplotlib.pyplot as plt

# Libs
import numpy as np
import pickle


def box_crossvalidation(xval):

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


if __name__ == '__main__':
    xval = pickle.load(open('xval.p', 'rb'))
    box_crossvalidation(xval)
