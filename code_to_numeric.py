import numpy as np

def code_to_numeric(labels):
    code_dict = {code:num for num, code in enumerate(np.unique(labels))}
    label_array = np.array(labels).reshape(labels.shape[0]).astype(str)
    numeric_list = [code_dict[label] for label in label_array]

    return numeric_list