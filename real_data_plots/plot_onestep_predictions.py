from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import collections, cPickle, os

import pyhsmm
from pyhsmm.util.text import progprint_xrange, progprint

lookaheads = range(1,60,1)
test_slice = slice(10000,12000)
test_data_cache_path = '/scratch/test_data_cache.pickle'

#############
#  Loading  #
#############

### models

print 'Loading models...'

models = collections.OrderedDict()

# GMM
with open('/scratch/gmm_results.pickle','r') as infile:
    models['GMMs'] = cPickle.load(infile)

# HMM
with open('/scratch/hmm_results.pickle','r') as infile:
    models['HMMs'] = cPickle.load(infile)

# HSMM
with open('/scratch/hsmm_results.pickle','r') as infile:
    models['HSMMs'] = cPickle.load(infile)

print '...done'

### data

print 'Loading data...'

if os.path.exists(test_data_cache_path):
    with open(test_data_cache_path,'r') as infile:
        test_data = cPickle.load(infile)
else:
    f = np.load('/scratch/TMT_50p_5-8-13_processed_notpca.npz')
    data = f['data']
    test_data = data[test_slice]
    with open(test_data_cache_path,'w') as outfile:
        cPickle.dump(test_data,outfile,protocol=-1)
    print 'saved to cache %s' % test_data_cache_path

print '...done'

#############
#  Running  #
#############

likelihoods = collections.OrderedDict()
for name, ms in progprint(models.iteritems()):
    likelihoods[name] = \
            np.asarray([m.predictive_likelihoods(test_data,lookaheads)
                for m in ms]).mean(0)

plt.figure()
gmm_likes = np.asarray(map(np.mean,likelihoods['GMMs']))
for name, ls in likelihoods.iteritems():
    plt.plot(lookaheads,np.asarray(map(np.mean,ls)) - gmm_likes,'x-',label=name[:-1])

    # means = np.asarray(map(np.mean,ls)) - gmm_likes
    # stds = np.asarray([np.std(l - g) for l,g in zip(ls,gmm_likes)])
    # plt.errorbar(lookaheads,means,stds,label=name[:-1])
plt.xlabel('steps into future')
plt.ylabel('log predictive likelihood')
plt.legend()

plt.show()

