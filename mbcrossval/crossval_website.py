# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa

import os
import shutil

# Libs
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import pickle
import numpy as np

# Local imports
from oggm import utils
from mbcrossval.crossval_plots import crossval_timeseries, crossval_histogram
from mbcrossval.crossval_plots import crossval_boxplot
from mbcrossval import mbcfg


def create_website():

    # setup jinja
    file_loader = FileSystemLoader(mbcfg.PATHS['jinjadir'])
    env = Environment(loader=file_loader)

    # different versions
    vdf = pd.DataFrame([], columns=['version', 'min_maj', 'file',
                                    'wd', 'pd'])

    for x in os.listdir(mbcfg.PATHS['storage_dir']):
        parts = x.split('_')
        if (parts[0] == 'xval') and (parts[2] == 'minor.p'):

            # make webdir
            webdir = os.path.join(mbcfg.PATHS['webroot'], parts[1], 'web')
            shutil.rmtree(webdir, ignore_errors=True)
            utils.mkdir(webdir)

            # check if plots do exist
            pltdir = os.path.join(mbcfg.PATHS['webroot'], parts[1], 'plots')
            mbcfg.PATHS['plotdir'] = pltdir

            if mbcfg.PARAMS['redo_all_plots']:
                utils.mkdir(pltdir)
                # try to make plots
                crossval_timeseries(os.path.join(mbcfg.PATHS['storage_dir'],
                                                 x))
                crossval_histogram(os.path.join(mbcfg.PATHS['storage_dir'], x))

            vdf = vdf.append({'version': parts[1],
                              'min_maj': parts[2].split('.')[0],
                              'file': os.path.join(mbcfg.PATHS['storage_dir'],
                                                   x),
                              'wd': webdir,
                              'verdir': os.path.join(mbcfg.PATHS['webroot'],
                                                     parts[1]),
                              'pd': mbcfg.PATHS['plotdir']},
                             ignore_index=True)
        elif (parts[0] == 'xval') and (mbcfg.PARAMS['redo_all_plots']) and\
                (parts[2] == 'major.p'):
            pltdir = os.path.join(mbcfg.PATHS['webroot'], parts[1], 'plots')
            mbcfg.PATHS['plotdir'] = pltdir

            utils.mkdir(pltdir)
            crossval_boxplot(os.path.join(mbcfg.PATHS['storage_dir'], x))

    vdf = vdf.sort_values(by='version')
    vdf.index = np.arange(len(vdf))

    for nr, vers in vdf.iterrows():

        # read data
        xvaldict = pickle.load(open(vers['file'], 'rb'))
        df = xvaldict['per_glacier']

        # sort array
        df.sort_values('Name', inplace=True)
        # move glaciers without name to the end
        df = pd.concat([df.loc[df.Name != ''], df.loc[df.Name == '']])
        # concatenate the overview to the beginning
        df = pd.concat([pd.DataFrame([{'Name': '',
                                       'RGIId': 'Overview',
                                       'xval_bias': df.xval_bias.mean(),
                                       'tstar_bias': df.tstar_bias.mean()}]),
                        df],
                       ignore_index=True)

        # set index ot RGIId
        df.index = df.RGIId

        #
        #
        # LINKNAMEs
        df['linkname'] = df.RGIId
        df.loc[df.Name != '', 'linkname'] = df.loc[df.Name != '', 'linkname']\
            + ', ' + df.loc[df.Name != '', 'Name']

        #
        # LINKLIST for GLACIERS
        df['link'] = df.RGIId + '.html'
        df.loc['Overview', 'link'] = '../index.html'
        template = env.get_template('createlinklist.txt')
        linklist = template.render(glaciers=df.to_dict(orient='records'))
        with open(os.path.join(mbcfg.PATHS['jinjadir'], 'linklist.html'),
                  'w') as fl:
            fl.write(linklist)

        #
        # LINKLIST for INDEX
        df['link'] = 'web/' + df.RGIId + '.html'
        df.loc['Overview', 'link'] = 'index.html'
        linklist = template.render(glaciers=df.to_dict(orient='records'))
        with open(os.path.join(mbcfg.PATHS['jinjadir'], 'linklistindex.html'),
                  'w') as fl:
            fl.write(linklist)

        #
        # WRITE ACTUAL HTML FILES
        template = env.get_template('template.html')
        fallback = env.get_template('fallback.html')
        for idx, glc in df.iterrows():

            crossval = 0

            # DIFFERENT VALUES DEPENDING ON INDEX OR GLACIER
            if glc.RGIId == 'Overview':
                # first: index
                bias1 = '  Mean t_star bias:'.ljust(27) + \
                        "{0:5.1f}".format(glc['tstar_bias'])
                bias2 = 'Mean crossval bias:'.ljust(27) + \
                        "{0:5.1f}".format(glc['xval_bias'])

                htmlname = os.path.join(vers['verdir'], 'index.html')
                imgname = 'plots/mb_histogram.png'
                index = 1
                linksuffix = '../'

                # path to where the crossval plots SHOULD BE stored
                cvplots = [x for x in os.listdir(vers['pd'])
                           if 'crossval' in x]
                if len(cvplots) > 0:
                    crossval = 1

            else:
                # second glaciers
                bias1 = '    Calibrated MB bias:'.ljust(35) + \
                        "{0:5.6f}".format(glc['tstar_bias'])
                bias2 = 'Crossvalidated MB bias:'.ljust(35) + \
                        "{0:5.1f}".format(glc['xval_bias'])

                htmlname = os.path.join(vers['wd'], glc['RGIId']) + '.html'
                imgname = '../plots/%s.png' % glc['RGIId']
                index = 0
                linksuffix = '../../'
                cvplots = ''

            #
            # Add PREVIOUS/NEXT buttons and link them
            if (vers == vdf.iloc[0]).all() & (len(vdf) > 1):
                # first version, no previous
                previous = ''
                nxtlink = os.path.join(linksuffix, vdf.iloc[nr+1]['version'],
                                       glc['link'])
                nxtfile = os.path.join(mbcfg.PATHS['webroot'],
                                       vdf.iloc[nr+1]['version'], glc['link'])

                next = '<a href="%s" class="next">%s &raquo;</a>' %\
                       (nxtlink, vdf.iloc[nr+1]['version'])
                if not os.path.isfile(nxtfile):
                    fbhtml = fallback.render()
                    with open(nxtfile, 'w') as fb:
                        fb.write(fbhtml)

            elif (vers == vdf.iloc[-1]).all() & (len(vdf) > 1):
                # last version, no next
                next = ''
                prvlink = os.path.join(linksuffix, vdf.iloc[nr-1]['version'],
                                       glc['link'])
                prvfile = os.path.join(mbcfg.PATHS['webroot'],
                                       vdf.iloc[nr-1]['version'], glc['link'])

                previous = '<a href="%s" class="previous">&laquo; %s</a>' % \
                           (prvlink, vdf.iloc[nr-1]['version'])
                if not os.path.isfile(prvfile):
                    fbhtml = fallback.render()
                    with open(prvfile, 'w') as fb:
                        fb.write(fbhtml)

            elif len(vdf) == 1:
                next = ''
                previous = ''
            else:
                nxtlink = os.path.join(linksuffix, vdf.iloc[nr+1]['version'],
                                       glc['link'])
                nxtfile = os.path.join(mbcfg.PATHS['webroot'],
                                       vdf.iloc[nr+1]['version'], glc['link'])
                prvlink = os.path.join(linksuffix, vdf.iloc[nr-1]['version'],
                                       glc['link'])
                prvfile = os.path.join(mbcfg.PATHS['webroot'],
                                       vdf.iloc[nr-1]['version'], glc['link'])

                previous = '<a href="%s" class="previous">&laquo; %s</a>' % \
                           (prvlink, vdf.iloc[nr-1]['version'])
                next = '<a href="%s" class="next">%s &raquo;</a>' % \
                       (nxtlink, vdf.iloc[nr+1]['version'])
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
                                      version=xvaldict['oggmversion'],
                                      date=xvaldict['date_created'],
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
        if (vers == vdf.iloc[-1]).all():
            template = env.get_template('latestindex.html')
            latest = template.render(version=vers.version)
            latestfile = os.path.join(mbcfg.PATHS['webroot'], 'index.html')

            with open(latestfile, 'w') as fl:
                fl.write(latest)
