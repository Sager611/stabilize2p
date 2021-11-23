import logging
from .main import *

__author__ = "Adrian Sager"
__email__ = "adrian.sagerlaganga@epfl.ch"


def initialize():
    if '_stabilizer2p_initialized' in globals():
        return
    global _ofco_initialized
    _stabilizer2p_initialized = True

    logger = logging.getLogger('stabilizer2p')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    formatter = \
        logging.Formatter('[%(asctime)s] %(levelname).1s T%(thread)d %(filename)s:%(lineno)s: %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


initialize()