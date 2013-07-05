from __future__ import division
import numpy as np
import cPickle

import pyhsmm
import library_models
from pyhsmm.util.text import progprint_xrange

# NOTE: this hierarchy of mixture models is totall unidentifiable and doesn't
# predict well because it can't look at temporal cues at all. that also means
# the hmm and the hsmm don't degrade into this model; they stay better.

num_iter = 200
training_slice = slice(0,10000)

#############
#  Loading  #
#############

### data

f = np.load('/scratch/TMT_50p_5-8-13_processed_notpca.npz')
# f = np.load('/Users/mattjj/Desktop/TMT_50p_5-8-13_processed_notpca.npz')
data = f['data']
mus = f['mu']
sigmas = f['sigma']
training_data = data[training_slice]

### library

library_size, obs_dim = mus.shape

component_library = \
        [pyhsmm.basic.distributions.GaussianFixed(
            mu=mu,sigma=sigma,
            ) for mu,sigma in zip(mus,sigmas)]

#################
#  Build model  #
#################

n_states = 200

init_weights = np.random.random((n_states,library_size))

obs_distns = [
        library_models.FrozenMixtureDistribution(
            components=component_library,
            a_0=1.0,b_0=1./5,weights=weights, # for initialization only
        ) for weights in init_weights]

p_prior, n_prior = 0.5, 100
alpha_0 = p_prior*n_prior
beta_0 = (1.0-p_prior)*n_prior
dur_distns = [pyhsmm.basic.distributions.NegativeBinomialIntegerRVariantDuration(
    np.ones((10,)),
    alpha_0=alpha_0,beta_0=beta_0) for state in range(n_states)]

hsmm = library_models.LibraryHSMMIntNegBinVariant(
        init_state_concentration=10.,
        alpha_a_0=1.0,alpha_b_0=1./10,
        gamma_a_0=1,gamma_b_0=1,
        # alpha=2, gamma=20.0,
        obs_distns=obs_distns,
        dur_distns=dur_distns)
hsmm.add_data(training_data)

##########################
#  Gather model samples  #
##########################

for itr in progprint_xrange(num_iter):
    hsmm.resample_model()

hsmms = [hsmm.resample_and_copy() for itr in progprint_xrange(1)]

##########
#  Save  #
##########

with open('/scratch/hsmm_results.pickle','w') as outfile:
    cPickle.dump(hsmms,outfile,protocol=-1)

