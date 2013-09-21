from __future__ import division
import numpy as np
import cPickle

import pyhsmm
import library_models
from pyhsmm.util.text import progprint_xrange

num_iter = 500
training_slice = slice(0,100000)

#############
#  Loading  #
#############

### data

# f = np.load('/scratch/TMT_6-3-13_median_7x3x3_zscore-norm_madeon_200libsize_8-23-2913-fororchestra.npz') # 200
f = np.load('/scratch/TMT_6-3-13_median_7x3x3_zscore-norm_madeon_600libsize_8-23-2913-fororchestra.npz')

data = f['data']
mus = f['means']
sigmas = f['sigmas']
training_data = data[training_slice]

### library

library_size, obs_dim = mus.shape

obs_distns = \
        [pyhsmm.basic.distributions.GaussianFixed(
            mu=mu,sigma=sigma,
            ) for mu,sigma in zip(mus,sigmas)]

#################
#  Build model  #
#################

hmm = library_models.FrozenHMM(
        alpha_a_0=1.0,alpha_b_0=0.1,
        gamma_a_0=1,gamma_b_0=1,
        obs_distns=obs_distns)
hmm.add_data(training_data)

##########################
#  Gather model samples  #
##########################

for itr in progprint_xrange(num_iter):
    hmm.resample_model()

##########
#  Save  #
##########

with open('/scratch/frozenhmm_results.pickle','w') as outfile:
    cPickle.dump(hmm,outfile,protocol=-1)

