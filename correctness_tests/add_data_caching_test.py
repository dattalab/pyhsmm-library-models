from __future__ import division
import numpy as np
import time

from pyhsmm.models import HSMMIntNegBinVariant
from pyhsmm.basic.models import MixtureDistribution
from library_models import FrozenMixtureDistribution, LibraryHSMMIntNegBinVariant
from pyhsmm.basic.distributions import Gaussian, NegativeBinomialIntegerRVariantDuration

#############################
#  generate synthetic data  #
#############################

states_in_hsmm = 5
components_per_GMM = 3
component_hyperparameters = dict(mu_0=np.zeros(200),sigma_0=np.eye(200),kappa_0=0.025,nu_0=220)

GMMs = [MixtureDistribution(
    alpha_0=4.,
    components=[Gaussian(**component_hyperparameters) for i in range(components_per_GMM)])
    for state in range(states_in_hsmm)]

true_dur_distns = [NegativeBinomialIntegerRVariantDuration(np.r_[0.,0,0,1,1,1,1,1],alpha_0=5.,beta_0=5.)
        for state in range(states_in_hsmm)]

truemodel = HSMMIntNegBinVariant(
        init_state_concentration=10.,
        alpha=6.,gamma=6.,
        obs_distns=GMMs,
        dur_distns=true_dur_distns)

data, truelabels = truemodel.generate(10000)

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
    alpha_0=4.,
    weights=row)
    for row in init_weights]

################
#  build HSMM  #
################

dur_distns = [NegativeBinomialIntegerRVariantDuration(np.r_[0.,0,0,1,1,1,1,1],alpha_0=5.,beta_0=5.)
        for state in range(library_size)]

model = LibraryHSMMIntNegBinVariant(
        init_state_concentration=10.,
        alpha=6.,gamma=6.,
        obs_distns=obs_distns,
        dur_distns=dur_distns)

#####################
#  add_data timing  #
#####################

print 'this one should be slower!'
tic = time.time()
model.add_data(data)
toc = time.time()
print '...done in %f seconds' % (toc-tic)
print ''

print 'this one sholud be faster!'
tic = time.time()
model.add_data(data)
toc = time.time()
print '...done in %f seconds' % (toc-tic)
print ''

