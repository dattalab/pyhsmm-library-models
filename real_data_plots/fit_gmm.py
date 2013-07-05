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
        [pyhsmm.basic.distributions.Gaussian(
            mu=mu,sigma=sigma,
            mu_0=np.zeros(obs_dim),sigma_0=np.eye(obs_dim), # dummies, not used
            nu_0=obs_dim+10,kappa_0=1., # more dummies
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

gmm = library_models.LibraryMM(
        a_0=1.,b_0=1./10,
        components=obs_distns)
gmm.add_data(training_data)

##########################
#  Gather model samples  #
##########################

for itr in progprint_xrange(num_iter):
    gmm.resample_model()

gmms = [gmm.resample_and_copy() for itr in progprint_xrange(1)]

##########
#  Save  #
##########

with open('/scratch/gmm_results.pickle','w') as outfile:
    cPickle.dump(gmms,outfile,protocol=-1)

