# ORCHESTRA
from __future__ import division


num_iter = 5
training_slice = slice(0,290000) # out of a possible 290000
test_slice = slice(0,70000)
THIS_PROFILE = "default"
Nmaxsuper=60

import numpy as np
import cPickle, os
from IPython.parallel import Client
import pyhsmm
import pyhsmm.parallel # To kick off the worker imports'
# pyhsmm.parallel.set_profile(THIS_PROFILE)
import pyhsmm_library_models.library_subhmm_models as library_subhmm_models
from pyhsmm.util.text import progprint_xrange
import socket


# Utility functions
def trim_model(model):
    model.obs_distnss = []
    for subhmm in model.HMMs:
        subhmm.obs_distns = []
        for s in subhmm.states_list:
            s.data = None
    superstateseq = np.concatenate(model.stateseqs)
    substateseqs = [np.concatenate([s.stateseq for s in hmm.states_list]) if len(hmm.states_list) > 0 else [] for hmm in model.HMMs ]
    substateseqs_by_time = np.concatenate([s.substates for s in model.states_list])
    transmats = [s.trans_matrix for s in model.states_list]
    model.states_list = []
    return model, superstateseq,substateseqs,substateseqs_by_time,transmats

def get_model_goodies(model):
    superstateseq = np.concatenate(model.stateseqs)
    substateseqs = [np.concatenate([s.stateseq for s in hmm.states_list]) if len(hmm.states_list) > 0 else [] for hmm in model.HMMs ]
    substateseqs_by_time = np.concatenate([s.substates for s in model.states_list])
    return superstateseq, substateseqs, substateseqs_by_time


#############
#  Loading  #
#############

### data

if os.path.isdir('/hms'):
    # f = np.load('/hms/scratch1/abw11/Data/TMT_6-3-13_median_7x3x3_zscore-norm_madeon_200libsize_8-23-2913-fororchestra.npz')
    f = np.load('/hms/scratch1/abw11/Data/TMT_6-3-13_median_7x3x3_zscore-norm_madeon_100libsize_Nonequery8-23-2913-fororchestra.npz')
    
else:
    # f = np.load("/scratch/TMT_6-3-13_median_7x3x3_zscore-norm_madeon_200libsize_8-23-2913-fororchestra.npz")
    f = np.load('/scratch/TMT_6-3-13_median_7x3x3_zscore-norm_madeon_100libsize_Nonequery8-23-2913-fororchestra.npz')

training_data = f['data']
test_data = f['data_test']

mus = f['means']
sigmas = f['sigmas']
training_data = training_data[training_slice]
test_data = test_data[test_slice]

### library
library_size, obs_dim = mus.shape
library = \
        [pyhsmm.basic.distributions.GaussianFixed(
            mu=mu,sigma=sigma) for mu,sigma in zip(mus,sigmas)]

#################
#  Build model  #
#################

avg_dur = 5
n_examples_r = 5
avg_r = 5
alpha_0 = n_examples_r*avg_dur
beta_0 = n_examples_r*avg_r

# p_prior, n_prior = 0.5, 100
# alpha_0 = p_prior*n_prior
# beta_0 = (1.0-p_prior)*n_prior
dur_distns = [pyhsmm.basic.distributions.NegativeBinomialIntegerRVariantDuration(
    np.r_[1,1,1,1,1,1,1,1,1,1,1,1,1],
    alpha_0=alpha_0,beta_0=beta_0) for state in range(Nmaxsuper)]

model = library_subhmm_models.HSMMIntNegBinVariantFrozenSubHMMs(
        alpha_a_0       = 1.0,
        alpha_b_0       = 0.01,
        gamma_a_0       = 1,
        gamma_b_0       = 1,
        # sub_alpha       = 1.0e-5,
        # sub_gamma       = 5,
        sub_alpha_a_0   = 1.0,
        sub_alpha_b_0   = 0.2,
        sub_gamma_a_0   = 1.0,
        sub_gamma_b_0   = 1.0,
        obs_distnss=[library]*Nmaxsuper,
        dur_distns=dur_distns)

from IPython.parallel import Client
dviews = Client(profile=THIS_PROFILE)[:]
n_clients = len(dviews)
# Set the clients up with the right paths within them
home_dir = os.path.expanduser("~")
dviews.execute("""
import sys
sys.path.append("{home_dir}/Code")
sys.path.append("{home_dir}/Code/luigi/")
sys.path.append("{home_dir}/Code/")
sys.path.append("{home_dir}/Code/pyhsmm_library_models/")
import pyhsmm_library_models.library_subhmm_models as library_subhmm_models
""".format(home_dir=home_dir))
dviews.push(dict(my_data={}))
pyhsmm.parallel.data_residency = {}
pyhsmm.parallel.costs = np.zeros(len(dviews))
# np.seterr(all='ignore') # put this in the execute thingy

print "Distributing data to %d clients...\n" % n_clients
all_data = np.array_split(training_data, n_clients)
for this_data in all_data:
    # model.add_data_parallel(this_data,left_censoring=True)
    model.add_data(training_data,left_censoring=True)

##########################
#  Gather model samples  #
##########################

print "Beginning our resampling"
likelihoods = []
heldout_likelihoods = []

if os.path.isdir('/hms'):
    outdir = '/hms/scratch1/abw11/frozen_subhmm/frozen_subhmm_%dNmaxsuper/' % Nmaxsuper
else:
    outdir = '/scratch/frozen_subhmm_%dNmaxsuper/' % Nmaxsuper
if not os.path.exists(outdir):
    os.makedirs(outdir)

idx = None
for itr in progprint_xrange(num_iter,perline=1):
    print "About to enter resample_model_parallel"
    # model.resample_model_parallel()
    model.resample_model()
    print "Resampled model, now getting likelihoods"
    loglike = model.log_likelihood()/len(training_data)
    likelihoods.append(loglike)
    if itr == num_iter-1:
        heldout_loglike = model.log_likelihood(test_data)/len(test_data)
        heldout_likelihoods.append(heldout_loglike)

    print "Train like: %f" % loglike

    # Save some information on the model. This is useful for us. 
    with open(os.path.join(outdir, "model_%d.pickle" % itr), "w") as f:
        superseqs, subseqs,subseqs_by_time = get_model_goodies(model)
        cPickle.dump({"superseqs":superseqs,
                      "subseqs":subseqs,
                      "subseqs_by_time":subseqs_by_time,
                      "likelihood":loglike}, f)

    # Print out some information for us as we train. 
    substate_usage = np.array([len(np.unique(s)) for s in subseqs]) #waste not want not, subseqs!
    superstate_counts = np.bincount(superseqs, minlength=Nmaxsuper)
    if idx == None: idx = np.argsort(substate_usage)[::-1]
    print np.vstack((substate_usage[idx],superstate_counts[idx]))


##########
#  Save  #
##########

outfile = os.path.join(outdir, 'parallel_frozen_subhmm_results.pickle')

with open(outfile,'w') as f:
        this_model, superseqs, subseqs,subseqs_by_time,transmats = trim_model(model)
        cPickle.dump({"model":this_model,
                        "superseqs":superseqs,
                        "subseqs":subseqs,
                        "subseqs_by_time":subseqs_by_time,
                        "likelihoods":likelihoods,
                        "heldout_likelihoods":heldout_likelihoods}, f, protocol=-1)
