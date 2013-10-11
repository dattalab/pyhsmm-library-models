from __future__ import division
import numpy as np
import cPickle

import pyhsmm
import library_models
from pyhsmm.util.text import progprint_xrange

num_iter = 50
training_slice = slice(0,35000)

#############
#  Loading  #
#############

### data

f = np.load('/scratch/TMT_25p_6-3-13_mm_median_7x3x3_zscore-norm_Truemousenormed_somlibtype_200libsize_Nonequery_9-30-2013.npz')

data = f['data']
mus = f['means']
sigmas = f['sigmas']
training_data = data[training_slice]

### library

library_size, obs_dim = mus.shape

library = \
        [pyhsmm.basic.distributions.GaussianFixed(
            mu=mu,sigma=sigma) for mu,sigma in zip(mus,sigmas)]

#################
#  Build model  #
#################

p_prior, n_prior = 0.5, 100
alpha_0 = p_prior*n_prior
beta_0 = (1.0-p_prior)*n_prior
dur_distns = [pyhsmm.basic.distributions.NegativeBinomialIntegerRVariantDuration(
    np.r_[0,0,1,1,1,1,1,1,1,1,1,1,1],
    alpha_0=alpha_0,beta_0=beta_0) for state in range(n_states)]

model = HSMMIntNegBinVariantFrozenSubHMMs(
        alpha_a_0=1.0,alpha_b_0=0.1,
        gamma_a_0=1,gamma_b_0=1,
        sub_alpha_a_0=1.,sub_alpha_b_0=1.,sub_gamma_a_0=1.,sub_gamma_b_0=1.,
        obs_distnss=[library]*Nmaxsuper,
        dur_distns=dur_distns)

model.add_data(training_data)

##########################
#  Gather model samples  #
##########################

for itr in progprint_xrange(num_iter):
    model.resample_model()

##########
#  Save  #
##########

with open('/scratch/frozen_subhmm_results.pickle','w') as outfile:
    cPickle.dump(model,outfile,protocol=-1)

