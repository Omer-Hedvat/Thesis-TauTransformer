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


def get_user_input(message_text, possible_answers=None):
    """Presents a message s and expects the user to give one of the optional possible answers. Returns the answer."""
    ans = input(message_text)
    if possible_answers is not None:
        while ans not in possible_answers:
            ans = input(message_text)

    return ans


def flatten(t):
    """
    given a matrix, returns a flatten list
    :param t:
    :return:
    """
    return [item for sublist in t for item in sublist]


def calc_mean_std(arr):
    """
    Calculates matrix's mean & std (of entire matrix)
    :return: mean, std
    """
    mean = arr.mean().mean()
    var = sum([((x - mean) ** 2) for x in flatten(arr.values)]) / len(flatten(arr.values))
    std = var ** 0.5
    return mean, std


def update_dict(tree, keys, value):
    """
    Through this function we create/update the results.json file in the right place for each metric
    :param tree: The dictionary to be updated (in-place).
    :param keys: Array of keys in the dictionary - path to where to add the value.
    :param value: The value to add. Can be any type.
    :return: None.
    """
    subtree = tree
    for key in keys[:-1]:
        if key not in subtree:
            subtree[key] = {}
        subtree = subtree[key]

    subtree[keys[-1]] = value


def lists_avg(lst):
    return sum(lst)/len(lst)


def calc_k(features, prc):
    return int(len(features) * prc)


def arrange_data_features(train_set, val_set, feature, return_y=True):
    if return_y:
        return train_set[feature].copy(), train_set['label'].copy(), val_set[feature].copy(), val_set['label'].copy()
    else:
        return train_set[feature].copy(), val_set[feature].copy()
