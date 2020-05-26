import numpy as np
import time
from utils.general_utils import loadpklz, savepklz

from multiprocessing import Pool, TimeoutError
import time
import os
import tqdm

import models

cmab = models.__dict__['Slplatoon_withContext']()

### enumerate all mu
args = np.meshgrid(range(cmab.nArms), range(cmab.nZ), range(cmab.nU))
def func(arg):
    context = arg[1:]
    arm = arg[0]
    reward = []
    np.random.seed(1024)
    for _ in range(mult):
        reward.append(cmab.play_context(arm, context))
    return np.mean(reward)

with Pool(processes=25) as pool:
    mu = list(tqdm.tqdm(pool.imap(func, args), total=len(initial_states)))

from IPython import embed; embed()
### simulate

### extract delta_z

### extract bounds

### UCB_SI
