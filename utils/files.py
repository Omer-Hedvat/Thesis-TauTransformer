import csv
import logging
import traceback

import numpy as np
import pandas as pd
import random
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
        ts = f"{datetime.now().strftime('%d%m%Y')}"
        resolved_path = f"{resolved_path}_{ts}"

    os.makedirs(resolved_path, exist_ok=True)

    if on_exists == 'ignore':
        # print(f"work_dir {resolved_path} already exists. Ignoring.")
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


def save_json(data, workdir, filename, indent=2, _jsonify=True):
    """
    Saves json data.
    :param data: a json-valid data object
    :param filename: a path to the file
    :param indent: indentation for the json
    :param _jsonify: whether to 'jsonify' (convert to json valid format) the input data
    :return: None.
    """
    import os
    import json
    import json5
    from functools import partial
    from pathlib import Path

    suffix = Path(filename).suffix
    assert suffix in ['.json', '.json5']

    if _jsonify:
        data = jsonify(data)

    dump = partial(json5.dump, quote_keys=True, trailing_commas=False) if suffix == '.json5' else json.dump
    with open(os.path.join(workdir, filename), 'wt', newline='\n') as F:
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
    from joblib import Parallel, delayed

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

    # Encode target column if needed
    if data[config['label_column']].dtype == 'object':
        le = preprocessing.LabelEncoder()
        data[config['label_column']] = le.fit_transform(data[config['label_column']])

        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        logger.info(f"Target column mapping: {le_name_mapping}")
    elif data[config['label_column']].dtype == 'float':
        data[config['label_column']] = data[config['label_column']].astype('int')

    # Change target column to 'label'
    if config['label_column'] != 'label':
        data.rename(columns={config['label_column']: 'label'}, inplace=True)

    # Add features if needed
    if config.get('add_features_up_to', 0) > data.shape[1]:
        additional_columns = config.get('add_features_up_to', 0) - data.shape[1]
        dummy_features_list = Parallel(n_jobs=-1)(
            delayed(generate_columns)(data)
            for _ in range(additional_columns)
        )
        dummy_feature_names = [f'dummy_feature_{i}' for i in range(additional_columns)]
        dummy_features = pd.concat(dummy_features_list, axis=1).set_axis(dummy_feature_names, axis=1)
        data = pd.concat([data, dummy_features], axis=1)
        config['dataset_name'] = f"{config['dataset_name']}_w_dummy_features_{config['add_features_up_to']}"

    return data, config['dataset_name']


def generate_columns(data):
    random_feature = random.randint(0, data.shape[1]-1)
    mean = data.describe().iloc[[1], random_feature].values[0]
    std = data.describe().iloc[[2], random_feature].values[0]
    col = pd.DataFrame(np.random.normal(loc=mean, scale=std, size=data.shape[0]))
    if data.dtypes[random_feature] == 'int64':
        col = col.astype(int)
    return col


def print_separation_dots(message):
    star_number = round((100 - len(message) - 2) / 2)
    logger.info('*' * 100)
    logger.info(f"{'*' * star_number} {message} {'*' * star_number}")
    logger.info('*' * 100)


def return_ds_results_mask(filename, dataset, features_prc):
    from datetime import datetime
    df = pd.read_csv(filename)
    today_date = datetime.now().strftime('%d-%m-%Y')
    ds_results_mask = (
            (df.dataset == dataset) & (df.features_prc == features_prc) & (df.date == today_date)
    )
    return df, ds_results_mask, today_date


def store_results(dataset, features_prc, metric, acc, f1, classes, workdir, timer_list=None):
    from datetime import datetime
    import os
    from utils.general import lists_avg
    from utils.machine_learning import calc_f1_score

    # General Results File
    filename = 'results/all_datasets_results.csv'
    acc_results_df, ds_results_mask, today_date = return_ds_results_mask(filename, dataset, features_prc)
    if ds_results_mask.any():
        acc_results_df.loc[ds_results_mask, metric] = round(lists_avg(acc), 3)
    else:
        new_df = pd.DataFrame(columns=acc_results_df.columns)
        new_df.loc[len(new_df), ['date', 'dataset', 'features_prc', metric]] = \
            [today_date, dataset, features_prc, round(lists_avg(acc), 3)]
        acc_results_df = pd.concat([acc_results_df, new_df]).sort_values(by=['dataset', 'features_prc'])
    acc_results_df.to_csv('results/all_datasets_results.csv', index=False)

    # Dataset's F1 Results File
    columns = ['features_prc', *[f'{metric}_{class_name}' for class_name in classes]]
    class_avg_f1 = calc_f1_score(f1)
    values = [features_prc, *class_avg_f1]
    data_dict = dict(zip(columns, values))
    f1_file = os.path.join(workdir, f'f1_scores.csv')
    new_data_df = pd.DataFrame([data_dict])
    if not os.path.exists(f1_file):
        new_data_df.to_csv(f1_file, index=False)
    else:
        f1_results_df = pd.read_csv(f1_file)
        all_ds_results_mask = ((f1_results_df.features_prc == features_prc))
        if all_ds_results_mask.any():
            f1_results_df.loc[all_ds_results_mask, columns] = values
        else:
            f1_results_df = pd.concat([f1_results_df, new_data_df]).sort_values(by=['features_prc'])
        f1_results_df.to_csv(f1_file, index=False)

    # Timer Results File
    filename = 'results/timer_results.csv'
    if timer_list:
        timer_avg = round(lists_avg([t.to_int() for t in timer_list]), 3)
        timer_df, ds_results_mask, today_date = return_ds_results_mask(filename, dataset, features_prc)
        if ds_results_mask.any():
            timer_df.loc[ds_results_mask, metric] = timer_avg
        else:
            new_df = pd.DataFrame(columns=timer_df.columns)
            new_df.loc[len(new_df), ['date', 'dataset', 'features_prc', metric]] = \
                [today_date, dataset, features_prc, timer_avg]
            timer_df = pd.concat([timer_df, new_df]).sort_values(by=['dataset', 'features_prc'])
        timer_df.to_csv('results/timer_results.csv', index=False)


def all_results_colorful():
    data = pd.read_csv("results/all_datasets_results.csv")
    dat = data['dataset'] + " " + data['features_prc'].apply(str) + " " + data['date'].apply(str)
    data['raw'] = dat
    data = data.set_index('raw')
    data = data.drop(columns=['date', 'dataset', 'features_prc'])
    data.style.background_gradient(cmap='RdYlGn', axis=1).to_excel("results/all_results_colors.xlsx")


def generate_and_save_scatter_plots(dm_dict, workdir=None):
    import matplotlib.pyplot as plt
    import os

    for dist_func, corr_dict in dm_dict.items():
        title = f"{dist_func}_DM1_scatter"
        data = corr_dict['coordinates']

        plt.scatter(data.T[:, 0], data.T[:, 1])
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")

        if workdir:
            path = os.path.join(workdir, 'scatter_plots')
            path = create_work_dir(path, append_timestamp=True, on_exists='ignore')
            file_path = os.path.join(path, title)
            plt.savefig(f'{file_path}.png')


def read_df_from_json(filename=None, json_data=None, attr=None):
    """
    takes a pandas dataframe as a string (from 'to_json()') back to a dataframe
    :param filename: filename with path
    :param json_data: json file
    :param attr: the dictionary's attribute name
    :return: a pd.DataFrame
    """
    import json

    if filename:
        with open(filename) as f_obj:
            json_data = json.load(f_obj)

    df = pd.read_json(json_data[attr]) if attr else pd.read_json(json_data)
    return df
