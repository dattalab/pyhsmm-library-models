from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

from pyhsmm.models import HSMMIntNegBinVariant
from pyhsmm.basic.models import MixtureDistribution
from library_models import FrozenMixtureDistribution, LibraryHSMMIntNegBinVariant
from pyhsmm.basic.distributions import Gaussian, NegativeBinomialIntegerRVariantDuration
from pyhsmm.util.text import progprint_xrange

#############################
#  generate synthetic data  #
#############################

states_in_hsmm = 5
components_per_GMM = 3
component_hyperparameters = dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.01,nu_0=3)

GMMs = [MixtureDistribution(
    alpha_0=4.,
    components=[Gaussian(**component_hyperparameters) for i in range(components_per_GMM)])
    for state in range(states_in_hsmm)]

true_dur_distns = [NegativeBinomialIntegerRVariantDuration(np.r_[0.,0,0,0,0,1,1,1],alpha_0=5.,beta_0=5.)
        for state in range(states_in_hsmm)]

truemodel = HSMMIntNegBinVariant(
        init_state_concentration=10.,
        alpha=6.,gamma=6.,
        obs_distns=GMMs,
        dur_distns=true_dur_distns)

training_datas = [truemodel.generate(10000)[0] for i in range(5)]

#####################################
#  set up FrozenMixture components  #
#####################################

# list of all Gaussians
component_library = [c for m in GMMs for c in m.components]
library_size = len(component_library)

# initialize weights to indicator on one component
init_weights = np.eye(library_size)

obs_distns = [FrozenMixtureDistribution(
    components=component_library,
    alpha_0=4,
    weights=row)
    for row in init_weights]

################
#  build HSMM  #
################

dur_distns = [NegativeBinomialIntegerRVariantDuration(np.r_[0.,0,0,0,0,1,1,1],alpha_0=5.,beta_0=5.)
        for state in range(library_size)]

model = LibraryHSMMIntNegBinVariant(
        init_state_concentration=10.,
        alpha=6.,gamma=6.,
        obs_distns=obs_distns,
        dur_distns=dur_distns)

for data in training_datas:
    model.add_data(data,left_censoring=True)

##################
#  infer things  #
##################

samples1 = [model.resample_and_copy() for i in progprint_xrange(1)]
samples2 = [model.resample_and_copy() for i in progprint_xrange(10)]
samples3 = [model.resample_and_copy() for i in progprint_xrange(100)]
samples4 = [model.resample_and_copy() for i in progprint_xrange(1000)]

import cPickle
with open('samples1','w') as outfile:
    cPickle.dump(samples1,outfile,protocol=-1)

with open('samples2','w') as outfile:
    cPickle.dump(samples2,outfile,protocol=-1)

with open('samples3','w') as outfile:
    cPickle.dump(samples3,outfile,protocol=-1)

with open('samples4','w') as outfile:
    cPickle.dump(samples3,outfile,protocol=-1)

