from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import collections, cPickle, os, copy

import pyhsmm
from pyhsmm.util.text import progprint_xrange, progprint

import library_models

num_iter = 50
training_slice = slice(0,10000)
training_data_cache_path = '/scratch/training_data_cache.pickle'

#############
#  Loading  #
#############

### data

print 'Loading data...'

if os.path.exists(training_data_cache_path):
    with open(training_data_cache_path,'r') as infile:
        training_data = cPickle.load(infile)
else:
    f = np.load('/scratch/TMT_50p_5-8-13_processed_notpca.npz')
    data = f['data']
    training_data = data[training_slice]
    with open(training_data_cache_path,'w') as outfile:
        cPickle.dump(training_data,outfile,protocol=-1)
    print 'saved to cache %s' % training_data_cache_path

print '...done'

### models

print 'Loading models...'

models = collections.OrderedDict()

# HSMM
with open('/scratch/hsmm_results.pickle','r') as infile:
    hsmm = cPickle.load(infile)[-1]

print '...done'

###############
#  Degrading  #
###############

### build an HMM and a GMM using those frozen syllables
### NOTE: or we could determinstically map the HSMM into an HMM

gmm = library_models.LibraryMMFixedObs(
        a_0=1.,b_0=1./10,
        components=hsmm.obs_distns)
gmm.add_data(training_data)

for itr in progprint_xrange(num_iter):
    gmm.resample_model()
gmms = [gmm.resample_and_copy() for itr in progprint_xrange(1)]

with open('/scratch/gmm_from_hsmm_results.pickle','w') as outfile:
    cPickle.dump(gmms,outfile,protocol=-1)


hmm = library_models.LibraryHMMFixedObs(
        init_state_concentration=10.,
        alpha_a_0=1.0,alpha_b_0=1./10,
        gamma_a_0=1,gamma_b_0=1,
        obs_distns = hsmm.obs_distns)
hmm.add_data(training_data)

for itr in progprint_xrange(num_iter):
    hmm.resample_model()
hmms = [hmm.resample_and_copy() for itr in progprint_xrange(1)]

with open('/scratch/hmm_from_hsmm_results.pickle','w') as outfile:
    cPickle.dump(hmms,outfile,protocol=-1)


hsmm = library_models.LibraryHSMMIntNegBinVariantFixedObs(
        init_state_concentration=10.,
        alpha_a_0=1.0,alpha_b_0=1./10,
        gamma_a_0=1,gamma_b_0=1,
        obs_distns=hsmm.obs_distns,
        dur_distns=hsmm.dur_distns)
hsmm.add_data(training_data)

for itr in progprint_xrange(num_iter):
    hsmm.resample_model()
hsmms = [hsmm.resample_and_copy() for itr in progprint_xrange(1)]

with open('/scratch/hsmm_from_hsmm_results.pickle','w') as outfile:
    cPickle.dump(hsmms,outfile,protocol=-1)

