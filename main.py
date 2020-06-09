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
def func(arg):
    context = arg[1:]
    arm = arg[0]
    reward = []
    np.random.seed(1024)
    mult = 30000
    for _ in range(mult):
        reward.append(cmab.play_context(arm, context))
    return np.mean(reward)

if os.path.exists('oracle.pklz'):
    data = loadpklz('oracle.pklz')
    args = data['args']
    mu = data['mu']
else:
    args = np.meshgrid(range(cmab.nArms), range(cmab.nZ), range(cmab.nU))
    args = np.stack([arg.reshape(-1) for arg in args]).T
    args = [args[i] for i in range(args.shape[0])]

    with Pool(processes=25) as pool:
        mu = list(tqdm.tqdm(pool.imap(func, args), total=len(args)))

    savepklz({'mu':mu, 'args':args}, 'oracle.pklz')

# import ipdb; ipdb.set_trace()
# func(args[0])
args = np.stack(args)
mu = np.array(mu)

def print_zu(z, u):
  idx_zu = np.logical_and(args[:,1] == z, args[:,2] == u)
  args_zu = args[idx_zu, :]
  mu_zu = mu[idx_zu]
  print('z=%d, u=%d'%(z, u))
  print(args_zu[:,0])
  print(mu_zu)
  print('-----')

print_zu(0,0)
print_zu(0,1)
print_zu(1,0)
print_zu(1,1)

from IPython import embed; embed()

### simulate
budget = 10000
args = np.stack(args)
mu = np.array(mu)
cmab = models.__dict__['Slplatoon_withContext']()
log = []
for t in range(budget):
    # import ipdb; ipdb.set_trace()
    context = cmab.sample_context()
    idx = np.logical_and(args[:,1] == context[0], args[:,2] == context[1])
    A = args[np.where(idx)[0][mu[idx].argmax()],:][0]
    Y = cmab.play_context(A, context)
    log.append([A, Y])

# from IPython import embed; embed()

### extract delta_z
z = 0
args_z = args[args[:,1] == z, :]
mu_z = mu[args[:,1] == z]

for u in range(cmab.nU):
    mu_star = mu_z[args_z[:,2] == u].max()
    k_star = mu_z[args_z[:,2] == u].max()
    sub.append()
    sup.append(mu_star)
### extract bounds

### UCB_SI
