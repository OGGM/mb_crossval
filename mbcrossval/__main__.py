# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa
import os
import argparse

# Local imports
from oggm import utils
from mbcrossval.run import run_main
from mbcrossval import mbcfg


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--storage',
                        type=str,
                        help='Path to storage directory')
    parser.add_argument('--webroot',
                        type=str,
                        help='Path to website directory')
    parser.add_argument('--workdir',
                        type=str,
                        help='Path to working directory')

    defaultcfg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'defaultconfig.cfg')

    parser.add_argument('--config',
                        type=str,
                        default=defaultcfg,
                        help='Optional path to configuration file. ' +
                        'Uses defaultconfig.cfg as default.')

    parser.add_argument('--histalp',
                        dest='histalp',
                        action='store_true',
                        help='Optional. Used without argument, will make a ' +
                             'HISTALP crossvalidation run.')

    parser.set_defaults(histalp=False)

    args = parser.parse_args()

    # run configuration file
    mbcfg.initialize(args.config)

    # HISTALP
    mbcfg.PARAMS['histalp'] = args.histalp
    if mbcfg.PARAMS['histalp']:
        mbcfg.PARAMS['oggmversion'] = mbcfg.PARAMS['oggmversion'] + '-histalp'

    # Working directory
    mbcfg.PATHS['working_dir'] = os.path.abspath(args.workdir)
    utils.mkdir(mbcfg.PATHS['working_dir'])

    # Storage directory
    mbcfg.PATHS['storage_dir'] = os.path.abspath(args.storage)
    utils.mkdir(mbcfg.PATHS['storage_dir'])

    # Website root directory
    mbcfg.PATHS['webroot'] = os.path.abspath(args.webroot)
    utils.mkdir(mbcfg.PATHS['webroot'])

    # Plotdir
    mbcfg.PATHS['plotdir'] = os.path.join(mbcfg.PATHS['webroot'],
                                          mbcfg.PARAMS['oggmversion'],
                                          'plots')
    utils.mkdir(mbcfg.PATHS['plotdir'])

    run_main()


if __name__ == '__main__':
    main()
