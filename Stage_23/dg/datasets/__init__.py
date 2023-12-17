from __future__ import absolute_import
import warnings

#from .train_v1_s3 import Train_v1_s3
#from .train_v1_s3_gt import Train_v1_s3_gt
from .train_v1_s3_r1 import Train_v1_s3_r1
from .train_v1_s3_r1_all import Train_v1_s3_r1_all
from .train_v1_s3_r import Train_v1_s3_r
from .train_v1_s3_bw import Train_v1_s3_bw
from .train_v1_s3_all_bw import Train_v1_s3_all_bw
#from .train_v1_s3_all import Train_v1_s3_all
#from .train_v1_s27 import Train_v1_s27
#from .train_v2_s3 import Train_v2_s3
#from .train_v6_s3 import Train_v6_s3
#from .train_v2_s27 import Train_v2_s27
#from .train_v1_s3_np import Train_v1_s3_np
#from .train_v1_s27_np import Train_v1_s27_np


__factory = {
    #'train_v1_s3': Train_v1_s3,
    #'train_v1_s3_gt': Train_v1_s3_gt,
    'train_v1_s3_r1': Train_v1_s3_r1,
    'train_v1_s3_r1_all': Train_v1_s3_r1_all,
    'train_v1_s3_r': Train_v1_s3_r,
    'train_v1_s3_bw': Train_v1_s3_bw,
    'train_v1_s3_all_bw': Train_v1_s3_all_bw,
    #'train_v1_s3_all': Train_v1_s3_all,
    #'train_v1_s27': Train_v1_s27,
    #'train_v2_s3': Train_v2_s3,
    #'train_v6_s3': Train_v6_s3,
    #'train_v2_s27': Train_v2_s27,
    #'train_v1_s3_np': Train_v1_s3_np,
   # 'train_v1_s27_np': Train_v1_s27_np
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
