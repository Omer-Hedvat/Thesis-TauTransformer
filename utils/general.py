import logging
from typing import Union

logger = logging.getLogger(__name__)


def setup_logger(config: Union[str, dict], log_filename: str = None):
    """
    Configures a logging.logger object based on the configuration in 'algo_common/config_files/logger_config.json' file.
    If 'log_filename' is not provided, will create a log file in the 'tmp' directory of the OS.
    """
    import tempfile
    from logging.config import dictConfig
    from utils.files import load_json

    if logger.hasHandlers():
        return

    if isinstance(config, str):
        config = load_json(config)

    assert isinstance(config, dict)

    if log_filename is not None:
        config['handlers']['file_handler'] = dict(**config['handlers']['default_file_handler'], filename=log_filename)
        config['root']['handlers'].append('file_handler')

    config['handlers']['default_file_handler']['filename'] = tempfile.NamedTemporaryFile(mode='wt', prefix='log_', suffix='.txt', delete=False).name

    dictConfig(config)

    logger.info(f"Default log file at: {config['handlers']['default_file_handler']['filename']}")
    if log_filename is not None:
        logger.info(f"Log file at: {config['handlers']['file_handler']['filename']}")


def flatten(t):
    """
    given a matrix, returns a flatten list
    :param t:
    :return:
    """
    return [item for sublist in t for item in sublist]



def calc_mean_std(df):
    """
    Calculates matrix's mean & std (of entire matrix)
    :return: mean, std
    """
    mean = df.mean().mean()
    var = sum([((x - mean) ** 2) for x in flatten(df.values)]) / len(flatten(df.values))
    std = var ** 0.5
    return mean, std