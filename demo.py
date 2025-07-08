from usa_visa.logger import logging
from usa_visa.exception import AydieException
import sys


logging.info("Welcome to my custom log")

try:
    a = 2/0
except Exception as e:
    raise AydieException(e, sys)