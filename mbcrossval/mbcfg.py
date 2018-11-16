import sys
import os
from configobj import ConfigObj, ConfigObjError
import logging
import io
import re


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
        cp = ConfigObj(substitute_env(cfgfile), file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Param file could not be parsed (%s): %s', cfgfile, e)
        sys.exit()

    # --- PARAMETER ---

    # RGI Version
    PARAMS['rgi_version'] = cp['rgi_version']

    # which RGI region to process
    PARAMS['region'] = cp['region']
    if PARAMS['region'] == 'None':
        PARAMS['region'] = None

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
    PARAMS['redo_all_plots'] = cp.as_bool('redo_all_plots')
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
    PATHS['working_dir'] = ''

    # Storage directory
    PATHS['storage_dir'] = ''

    # Website root directory
    PATHS['webroot'] = ''

    # Plotdir
    PATHS['plotdir'] = ''

    # directory where jinja templates are stored, could be user defined...
    PATHS['jinjadir'] = cp['jinjadir']
    # ...or the default templates are used:
    if PATHS['jinjadir'] == 'None':
        jinjadir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'jinja_templates')
        PATHS['jinjadir'] = jinjadir


def substitute_env(filename):
    """Reads filename, substitutes environment variables and returns a file-like
     object of the result.

    Substitution maps text like "$FOO" for the environment variable "FOO".

    Taken from:
    https://github.com/aellerton
    https://github.com/DiffSK/configobj/issues/144#issuecomment-347019778
    """

    def lookup(match):
        """Replaces a match like $FOO with the env var FOO.
        """
        key = match.group(2)
        if key not in os.environ:
            raise Exception("Config env var not set")  # or ignore
        return os.environ.get(key)

    pattern = re.compile(r'(\$(\w+))')
    with open(filename, 'r') as src:
        content = src.read()
        replaced = pattern.sub(lookup, content)

    return io.StringIO(replaced)
