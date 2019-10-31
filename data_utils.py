import numpy as np
import pandas as pd


def code_to_numeric(labels):
    code_dict = create_code_dict(labels)
    label_array = np.array(labels).reshape(labels.shape[0]).astype(str)
    numeric_list = [code_dict[label] for label in label_array]

    return numeric_list


def create_code_dict(labels):
    unique_codes = np.unique(labels)
    code_dict = {code: num for num, code in enumerate(unique_codes)}

    return code_dict


def numeric_to_code(predictions, code_dict):
    code_list = []
    for id, pred in enumerate(predictions):
        code_string = get_keys_by_value(code_dict, pred)
        code_list.append(code_string[0])

    return code_list


def get_keys_by_value(dict, value):
    list_of_keys = []
    list_of_items = dict.items()
    list_of_items = dict.items()
    for item in list_of_items:
        if item[1] == value:
            list_of_keys.append(item[0])
    return list_of_keys


def save_to_csv(file_name, prediction_list):
    prediction_result = pd.DataFrame()
    prediction_result['ID'] = [i for i in range(1,1253)]
    prediction_result['Population'] = prediction_list
    prediction_result.to_csv(file_name,encoding='utf-8', index=False)
    return
