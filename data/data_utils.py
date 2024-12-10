import json

def load_dataset(data_path):
    """_summary_

    Args:
        data_path (_type_): _description_
    """
    with open(data_path, 'r') as data_file:
        dataset_dict = json.load(data_file)
    return dataset_dict


def filter_by_length(row):
    """_summary_

    Args:
        row (_type_): _description_
    """
    return len(row["text"])<1024