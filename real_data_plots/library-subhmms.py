from __future__ import division
import numpy as np
import cPickle

import pyhsmm
import library_subhmm_models
from pyhsmm.util.text import progprint_xrange


num_iter = 50
training_slice = slice(0,20000)

#############
#  Loading  #
#############

### data

f = np.load('/hms/scratch1/abw11/Data/TMT_6-3-13_median_7x3x3_zscore-norm_madeon_200libsize_8-23-2913-fororchestra.npz')

print "Loading data"
data = f['data']
mus = f['means']
sigmas = f['sigmas']
training_data = data[training_slice]

### library

library_size, obs_dim = mus.shape

print "Creating libraries"
library = \
        [pyhsmm.basic.distributions.GaussianFixed(
            mu=mu,sigma=sigma) for mu,sigma in zip(mus,sigmas)]

#################
#  Build model  #
#################

print "Preparing model"
Nmaxsuper=20
p_prior, n_prior = 0.5, 100
alpha_0 = p_prior*n_prior
beta_0 = (1.0-p_prior)*n_prior
dur_distns = [pyhsmm.basic.distributions.NegativeBinomialIntegerRVariantDuration(
    np.r_[0,0,1,1,1,1,1,1,1,1,1,1,1],
    alpha_0=alpha_0,beta_0=beta_0) for state in range(Nmaxsuper)]

model = library_subhmm_models.HSMMIntNegBinVariantFrozenSubHMMs(
        alpha_a_0=1.0,alpha_b_0=0.1,
        gamma_a_0=1,gamma_b_0=1,
        sub_alpha_a_0=1.,sub_alpha_b_0=1.,sub_gamma_a_0=1.,sub_gamma_b_0=1.,
        obs_distnss=[library]*Nmaxsuper,
        dur_distns=dur_distns)

print "Adding data, computing cached likelihoods"
model.add_data(training_data)

##########################
#  Gather model samples  #
##########################

print "Beginning resampling"
for itr in progprint_xrange(num_iter,perline=1):
    model.resample_model()

##########
#  Save  #
##########

with open('/hms/scratch1/abw11/frozen_subhmm_results.pickle','w') as outfile:
    cPickle.dump(model,outfile,protocol=-1)

