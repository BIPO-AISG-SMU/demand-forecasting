"""This package contains modules pertaining to differing parts
of the end-to-end workflow, excluding the source for serving
the model through a REST API."""
import sys

# Make sure old cache is not used
sys.dont_write_bytecode = False

# used to access the current decimal context, which provides control over precision settings.
from decimal import getcontext

getcontext().prec = 3
