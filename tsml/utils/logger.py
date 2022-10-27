from datetime import datetime
from email.policy import default
import functools
import logging

logger = logging.getLogger('TSML')
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s | %(message)s')
stream_handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)


def log_execution(_func=None, *, name='', logger = logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            f_name = name if name else func.__name__
            logger.info(f'Executing {f_name}')
            tic = datetime.now()
            result = func(*args, **kwargs)
            delta = round((datetime.now() - tic).total_seconds(), 2)
            logger.info(f'Finished {f_name} ({delta}) s')
            return result
        return wrapper
    if _func is None:
        return decorator
    else:
        return decorator(_func)
