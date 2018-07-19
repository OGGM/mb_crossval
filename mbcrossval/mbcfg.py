import sys
import os
from configobj import ConfigObj, ConfigObjError
import logging

import oggm
from oggm import utils
from oggm.cfg import ResettingOrderedDict
from oggm.cfg import PathOrderedDict

# Local logger
log = logging.getLogger(__name__)


# Globals
PARAMS = ResettingOrderedDict()
PATHS = PathOrderedDict()


def initialize(cfgfile):
    """Read the configuration file containing the run's parameters."""

    global PARAMS
    global PATHS

    log.info('Parameter file: %s', cfgfile)

    try:
        cp = ConfigObj(cfgfile, file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Param file could not be parsed (%s): %s', cfgfile, e)
        sys.exit()

    # --- PARAMETER ---

    # RGI Version
    PARAMS['rgi_version'] = cp['rgi_version']

    # which RGI region to process
    PARAMS['region'] = cp['region']

    # which climate file to use
    PARAMS['histalp'] = cp.as_bool('histalp')

    # OGGM Version
    if PARAMS['histalp']:
        PARAMS['oggmversion'] = oggm.__version__ + '-histalp'
    else:
        PARAMS['oggmversion'] = oggm.__version__

    # run tasks
    PARAMS['run_major_crossval'] = cp.as_bool('run_major_crossval')
    PARAMS['run_minor_crossval'] = cp.as_bool('run_minor_crossval')
    PARAMS['make_minor_plots'] = cp.as_bool('make_minor_plots')
    PARAMS['make_major_plots'] = cp.as_bool('make_major_plots')
    PARAMS['make_website'] = cp.as_bool('make_website')

    # major cross validation parameters
    PARAMS['prcp1'] = cp.as_float('prcp1')
    PARAMS['prcp2'] = cp.as_float('prcp2')
    PARAMS['prcp_step'] = cp.as_float('prcp_step')
    PARAMS['tliq1'] = cp.as_float('tliq1')
    PARAMS['tliq2'] = cp.as_float('tliq2')
    PARAMS['tliq_step'] = cp.as_float('tliq_step')
    PARAMS['tmel1'] = cp.as_float('tmel1')
    PARAMS['tmel2'] = cp.as_float('tmel2')
    PARAMS['tmel_step'] = cp.as_float('tmel_step')
    PARAMS['tgra1'] = cp.as_float('tgra1')
    PARAMS['tgra2'] = cp.as_float('tgra2')
    PARAMS['tgra_step'] = cp.as_float('tgra_step')

    # --- PATHS ---
    # OGGM working directory
    PATHS['working_dir'] = cp['working_dir']
    utils.mkdir(PATHS['working_dir'])

    # Storage directory
    PATHS['storage_dir'] = cp['storage_dir']
    utils.mkdir(PATHS['storage_dir'])

    # Website root directory
    PATHS['webroot'] = cp['webroot']
    utils.mkdir(PATHS['webroot'])

    # Plotdir
    PATHS['plotdir'] = os.path.join(PATHS['webroot'],
                                    PARAMS['oggmversion'],
                                    'plots')
    utils.mkdir(PATHS['plotdir'])

    # directory where jinja templates are stored
    PATHS['jinjadir'] = cp['jinjadir']

