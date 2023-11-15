from .glue_data_manager import GLUEDataManager

glue_tasks = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']

def get_dm_class(dataset_name):
    if dataset_name in glue_tasks:
        return GLUEDataManager
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))
