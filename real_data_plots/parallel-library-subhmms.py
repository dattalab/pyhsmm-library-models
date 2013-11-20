# JEFFERSON
from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nmaxsuper', metavar='N', type=int, nargs='+',
                    default=[10],
                   help='Number of super states')
parser.add_argument('--numiter', metavar='i', type=int, nargs='+',
                    default=[30],
                   help='Number of iterations')
parser.add_argument('--numframes', metavar='t', type=int, nargs='+',
                    default=[290000],
                   help='Number of frames to train on')
parser.add_argument('--numframestest', metavar='e', type=int, nargs='+',
                    default=[70000],
                   help='Number of frames to test on')
parser.add_argument('--numtestsplits', metavar='p', type=int, nargs='+',
                    default=[4],
                   help='Split held-out likelihood into a number of parts')
parser.add_argument('--profile', metavar='p', type=str, nargs='+',
                    default="default",
                   help='IPython parallel profile to use')


args = parser.parse_args()
Nmaxsuper = args.nmaxsuper[0]
num_iter = args.numiter[0]
T = args.numframes[0]
T_test = args.numframestest[0]
N_TEST_SPLITS = args.numtestsplits[0]
THIS_PROFILE = args.profile[0]
print THIS_PROFILE
profile = args.profile
training_slice = slice(0,T)
test_slice = slice(0,T_test)

import numpy as np
import cPickle, os
from IPython.parallel import Client
import pyhsmm
import pyhsmm.parallel # To kick off the worker imports'
pyhsmm.parallel.set_profile(THIS_PROFILE)
pyhsmm.parallel.reset_engines()
import pyhsmm_library_models.library_subhmm_models as library_subhmm_models
from pyhsmm_library_models.util import split_data
from pyhsmm.util.text import progprint_xrange
import time
import hashlib

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
elif os.path.isdir("/scratch/global"):
    f = np.load("/scratch/global/alexbw/Data/TMT_6-3-13_median_7x3x3_zscore-norm_madeon_200libsize_8-23-2913-fororchestra.npz")
    # f = np.load("/scratch/global/alexbw/Data/TMT_6-3-13_median_7x3x3_zscore-norm_madeon_100libsize_Nonequery8-23-2913-fororchestra.npz")
elif os.path.isdir("/scratch/dattalab"):
    f = np.load("/scratch/dattalab/TMT_6-3-13_median_7x3x3_zscore-norm_madeon_200libsize_8-23-2913-fororchestra.npz")
    # f = np.load("/scratch/dattalab/TMT_6-3-13_median_7x3x3_zscore-norm_madeon_100libsize_Nonequery8-23-2913-fororchestra.npz")
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
        sub_alpha_a_0   = 1.0e-5, # sub_alpha_a_0 and sub_alpha_b_0 most directly set substate counts
        sub_alpha_b_0   = 1, # mean is always a_0/b_0. When a_0 = 1.0, the distribution is exponential
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
all_data, all_aBl = split_data(training_data, model, n_clients)
print "The data is now split"
for i,(this_data,this_aBl) in enumerate(zip(all_data,all_aBl)):
    print "Now sending %d/%d" % (i,len(all_data))
    model.add_data_parallel(
        data=this_data,
        frozen_aBl=this_aBl,
        left_censoring=True,
        stateseq=np.zeros(this_data.shape[0]), # dummy (TODO speed up csr construction)
        )

##########################
#  Gather model samples  #
##########################

print "Beginning our resampling"
likelihoods = []
heldout_likelihoods = []


suffix = hashlib.md5(str(time.time())).hexdigest()[:6]
if os.path.isdir('/hms'):
    outdir = '/hms/scratch1/abw11/frozen_subhmm/frozen_subhmm_%dNmaxsuper_%s/' % (Nmaxsuper, suffix)
elif os.path.isdir("/scratch/global"):
    outdir = "/scratch/global/alexbw/frozen_subhmm/frozen_subhmm_%dNmaxsuper_%s/" % (Nmaxsuper, suffix)
elif os.path.isdir("/scratch/dattalab"):
    outdir = "/scratch/dattalab/frozen_subhmm/frozen_subhmm_%dNmaxsuper_%s/" % (Nmaxsuper, suffix)
else:
    outdir = '/scratch/frozen_subhmm_%dNmaxsuper_%s/' % (Nmaxsuper, suffix)

if not os.path.exists(outdir):
    os.makedirs(outdir)

idx = None
for itr in progprint_xrange(num_iter,perline=1):
    print "Resampling model..."
    model.resample_model_parallel()
    print "Resampled model, now getting likelihoods"
    loglike = model.log_likelihood()/len(training_data)
    likelihoods.append(loglike)
    if itr == num_iter-1:
        all_test_data = np.array_split(test_data, N_TEST_SPLITS)
        heldout_loglike_pieces = []
        for this_test_data in all_test_data:
            heldout_loglike_pieces.append(model.log_likelihood(this_test_data)/len(this_test_data))
        heldout_loglike = np.mean(heldout_loglike_pieces)
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
