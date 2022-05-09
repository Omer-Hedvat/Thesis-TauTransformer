import csv
import logging
import traceback

import numpy as np
import pandas as pd
from sklearn import preprocessing

from utils.timer import Timer

logger = logging.getLogger(__name__)


def create_work_dir(path, append_timestamp=False, on_exists='ask'):
    """
    Creates work directory in 'path'. If it already exists , checks 'on_exist' argument:
    :param path: The path we want to create
    :param append_timestamp: If true, adds the current timestamp to the 'path' argument.
    :param on_exists:
        if 'remove' - silently remove the existing folder and create a new one.
        if 'ignore' - won't remove the existing folder. New data will either be added to it or overwrite existing content.
        if 'abort' - will abort automatically,
        if 'ask' - will ask the user to choose - remove, ignore or abort.
        if 'raise' - will raise an exception
    :return path with its environment variables resolved.
            If directory creation failed, behaves as described above (no status is returned).
    """
    assert on_exists in ['ignore', 'ask', 'remove', 'abort', 'raise']

    from datetime import datetime
    import os
    import shutil
    from utils.general import get_user_input

    resolved_path = os.path.expandvars(path)
    assert {'$', '%'}.isdisjoint(resolved_path), f"Unrecognized environment variables exist in path '{resolved_path}'"

    if append_timestamp:
        resolved_path = os.path.normpath(resolved_path)
        ts = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        resolved_path = f"{resolved_path}_{ts}"

    os.makedirs(resolved_path, exist_ok=True)

    if on_exists == 'ignore':
        print(f"work_dir {resolved_path} already exists. Ignoring.")
        return resolved_path
    elif on_exists == 'abort':
        print(f"work_dir {resolved_path} already exists. Aborting.")
        exit(0)
    elif on_exists == 'ask':
        ans = get_user_input(
            message_text=f"work_dir {resolved_path} already exists: (r)emove/(i)gnore/(a)bort? ",
            possible_answers=['r', 'i', 'a']
        )
        if ans == 'r':
            shutil.rmtree(resolved_path)
            os.makedirs(resolved_path)
        elif ans == 'i':
            return resolved_path
        else:
            exit(0)
    elif on_exists == 'remove':
        print(f"work_dir {resolved_path} already exists. Removing.")
        shutil.rmtree(resolved_path)
        os.makedirs(resolved_path)
    elif on_exists == 'raise':
        raise FileExistsError(f"work_dir {resolved_path} already exists.")

    return resolved_path


def load_json(filename):
    """
    Loads json data. If the file extension is 'json5', it is loaded as json5 (for json5 features see https://json5.org)
    :param filename: a path to the file. Must be a valid json (or json5) text file.
    :return: the loaded data.
    """
    from collections import OrderedDict
    from pathlib import Path

    suffix = Path(filename).suffix
    assert suffix in ['.json', '.json5']

    if suffix == '.json5':
        import json5 as json
    else:
        import json

    with open(filename, 'rt') as F:
        data = json.load(F, object_pairs_hook=OrderedDict)

    return data


def save_json(data, filename, indent=2, _jsonify=True):
    """
    Saves json data.
    :param data: a json-valid data object
    :param filename: a path to the file
    :param indent: indentation for the json
    :param _jsonify: whether to 'jsonify' (convert to json valid format) the input data
    :return: None.
    """
    import json
    from functools import partial
    from pathlib import Path
    import json5

    suffix = Path(filename).suffix
    assert suffix in ['.json', '.json5']

    if _jsonify:
        data = jsonify(data)

    dump = partial(json5.dump, quote_keys=True, trailing_commas=False) if suffix == '.json5' else json.dump
    with open(filename, 'wt', newline='\n') as F:
        dump(data, F, indent=indent)


def load_pickle(filename):
    """
    Loads pickle data.
    :param filename: a path to the file.
    :return: the loaded data.
    """
    import pickle

    data = None
    try:
        # logger.info(f"Loading pickle file from '{filename}'.")
        with open(filename, 'rb') as F:
            data = pickle.load(F)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Failure loading pickle file. Error %s\n", tb)
        raise e

    return data


def save_pickle(data, filename):
    """
    Saves pickle data.
    :param data: a data object to save.
    :param filename: a path to the file.
    :return: None.
    """
    import pickle

    with open(filename, 'wb') as F:
        pickle.dump(data, F)


def jsonify(data, fix_non_string_dict_keys=False, max_float_decimals=4):
    """
    Convert possibly non json-compatible data types such as numpy arrays to a valid (json-wise) objects.
    """
    import datetime
    import json
    from collections import OrderedDict

    import numpy as np
    import pandas as pd

    if isinstance(data, dict):
        # Check for non-string keys
        non_string_keys = [key for key in data.keys() if not isinstance(key, str)]
        n_non_string_keys = len(non_string_keys)
        if n_non_string_keys > 0 and not fix_non_string_dict_keys:
            raise AssertionError(
                f"{n_non_string_keys} non-string keys in dictionary (e.g. '{non_string_keys[0]}'); "
                f"set 'fix_non_string_dict_keys' to fix automatically"
            )
        # Support OrderedDict (which extends dict) and native dict
        items = [(str(key), jsonify(value, fix_non_string_dict_keys)) for key, value in data.items()]
        return OrderedDict(items) if isinstance(data, OrderedDict) else dict(items)
    elif isinstance(data, (list, tuple, set, pd.Series, np.ndarray)):
        return [jsonify(item, fix_non_string_dict_keys) for item in data]
    elif isinstance(data, (float, np.float32, np.float64)):
        return round(float(data), max_float_decimals)
    elif isinstance(data, (np.uint8, np.uint16, np.uint32, np.int16, np.int32, np.int64)):
        return int(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, (datetime.date, datetime.time, datetime.datetime, datetime.timedelta)):
        return str(data)
    else:
        # Check for valid (json-wise) data by silently converting it to a JSON string. Will fail with the right error message if 'data' is not
        # JSON-valid.
        json.dumps(data)
        return data


def update_json_file(filename, keys, value):
    """
    Through this function we create/update the results.json file in the right place for each metric
    :param filename: Name of the JSON file. If it doesn't exist, it will be created.
    :param keys: Array of keys in the dictionary - path to where to add the value.
    :param value: The value to add. Can be any type.
    :return: None.
    """
    import os
    from utils.general import update_dict

    tree = load_json(filename) if os.path.exists(filename) else {}
    update_dict(tree, keys, value)
    save_json(tree, filename)


def read_from_csv(filepath, config):
    with open(filepath, "r", encoding="utf-8") as f, Timer() as timer:
        reader = csv.reader(f, delimiter=",")
        data = list(reader)
        nlinesfile = len(data)
    print(timer.to_string())

    if nlinesfile > config.get('nrows', -1) > 0:
        np.random.seed(0)
        lines2skip = np.random.choice(np.arange(1, nlinesfile + 1), (nlinesfile - config['nrows']), replace=False)
        data = pd.read_csv(filepath, skiprows=lines2skip)
    else:
        data = pd.read_csv(filepath)

    # Encode label column if needed
    if data[config['label_column']].dtype == 'object':
        le = preprocessing.LabelEncoder()
        data[config['label_column']] = le.fit_transform(data[config['label_column']])

        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        logger.info(f"Target column mapping: {le_name_mapping}")
    elif data[config['label_column']].dtype == 'float':
        data[config['label_column']] = data[config['label_column']].astype('int')

    if config['label_column'] != 'label':
        data.rename(columns={config['label_column']: 'label'}, inplace=True)

    return data


def print_separation_dots(message):
    star_number = round((100 - len(message) - 2)/2)
    logger.info('*' * 100)
    logger.info(f"{'*' * star_number} {message} {'*' * star_number}")
    logger.info('*' * 100)
