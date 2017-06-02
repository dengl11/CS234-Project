import numpy as np
from IPython.display import HTML


def sample_dict(dic, sample_size=5):
    keys = dic.keys()
    print("keys = {}".format(len(keys)))
    print("Sample of dict:")
    random_keys = np.random.choice(list(keys), sample_size, replace=False)
    for k in random_keys:
        print("- {}: {}".format(k, dic[k]))