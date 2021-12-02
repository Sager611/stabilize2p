import logging

__author__ = "Adrian Sager"
__email__ = "adrian.sagerlaganga@epfl.ch"


def initialize():
    if '_stabilize2p_initialized' in globals():
        return
    global _stabilize2p_initialized
    _stabilize2p_initialized = True

    logger = logging.getLogger('stabilize2p')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    formatter = \
        logging.Formatter('[%(asctime)s] %(levelname).1s T%(thread)d %(filename)s:%(lineno)s: %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


initialize()