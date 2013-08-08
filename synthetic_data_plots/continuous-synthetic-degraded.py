from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import collections

from pyhsmm.models import HSMMIntNegBinVariant
from pyhsmm.basic.models import MixtureDistribution
from pyhsmm.basic.distributions import Gaussian, NegativeBinomialIntegerRVariantDuration
from pyhsmm.util.text import progprint_xrange

from library_models import FrozenMixtureDistribution, LibraryHSMMIntNegBinVariant,\
        LibraryHMMFixedObs, LibraryMMFixedObs

resample_iter = 100
num_training_seqs = 3

#############################
#  generate synthetic data  #
#############################

states_in_hsmm = 5
components_per_GMM = 3
component_hyperparameters = dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.025,nu_0=3)

GMMs = [MixtureDistribution(
    alpha_0=4.,
    components=[Gaussian(**component_hyperparameters) for i in range(components_per_GMM)])
    for state in range(states_in_hsmm)]

true_dur_distns = [NegativeBinomialIntegerRVariantDuration(np.r_[0.,0,0,0,0,0,1,1,1,1],alpha_0=5.,beta_0=5.)
        for state in range(states_in_hsmm)]

truemodel = HSMMIntNegBinVariant(
        init_state_concentration=10.,
        alpha=6.,gamma=2.,
        obs_distns=GMMs,
        dur_distns=true_dur_distns)

training_datas = [truemodel.rvs(1000) for i in range(num_training_seqs)]
test_data = truemodel.rvs(1000)

#####################################
#  set up FrozenMixture components  #
#####################################

# list of all Gaussians
component_library = [c for m in GMMs for c in m.components]
library_size = len(component_library)

# initialize weights to indicator on one component
# NOTE: number of states does not need to be the same as the library size!
init_weights = np.eye(library_size)

obs_distns = [FrozenMixtureDistribution(
    components=component_library,
    alpha_0=4.,
    weights=row)
    for row in init_weights]

################
#  fit models  #
################

models = collections.OrderedDict()

### HSMM

dur_distns = [NegativeBinomialIntegerRVariantDuration(np.r_[0.,0,0,0,0,0,1,1,1,1],alpha_0=5.,beta_0=5.)
        for state in range(library_size)]

hsmm = LibraryHSMMIntNegBinVariant(
        init_state_concentration=10.,
        alpha=6.,gamma=2.,
        obs_distns=obs_distns,
        dur_distns=dur_distns)
for data in training_datas:
    hsmm.add_data(data,left_censoring=True)

for itr in progprint_xrange(resample_iter):
    hsmm.resample_model()

### degrade into HMM, use the same learned syllables!

hmm = LibraryHMMFixedObs(
        init_state_concentration=10.,
        alpha=6.,gamma=2.,
        obs_distns=hsmm.obs_distns)
for data in training_datas:
    hmm.add_data(data)

for itr in progprint_xrange(resample_iter):
    hmm.resample_model()

### degrade into GMM, use the same learned syllables!

gmm = LibraryMMFixedObs(
        a_0=1.,b_0=1./10,
        components=hsmm.obs_distns)
for data in training_datas:
    gmm.add_data(data)

for itr in progprint_xrange(resample_iter):
    gmm.resample_model()

##########
#  plot  #
##########

indices = range(1,100,1)

likelihoods = collections.OrderedDict((
        ('HSMM',hsmm.predictive_likelihoods(test_data,indices,left_censoring=True)),
        ('HMM',hmm.predictive_likelihoods(test_data,indices)),
        ('GMM',gmm.predictive_likelihoods(test_data,indices)),
        ))

plt.figure()
for name, ls in likelihoods.iteritems():
    plt.plot(indices,[np.mean(l) for l in ls],label=name)
plt.xlabel('time into future')
plt.ylabel('predictive likelihood')
plt.legend()

plt.show()

