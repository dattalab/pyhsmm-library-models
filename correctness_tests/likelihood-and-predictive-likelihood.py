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

training_datas = [truemodel.generate(1000)[0] for i in range(5)]
test_data = truemodel.generate(5000)[0]

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
    # model.add_data_parallel(data,left_censoring=True)

##################
#  infer things  #
##################

for i in progprint_xrange(25):
    model.resample_model()


#################
#  check likes  #
#################

computed_directly = model.log_likelihood(test_data,left_censoring=True)

# NOTE: this is like model.predictive_likelihoods(test_data,[1]) but it includes
# the first frame p(y_1) term instead of just starting at p(y_2|y_1)
s = model._states_class(model=model,data=test_data,stateseq=np.zeros(len(test_data)),left_censoring=True)
alphal = s.messages_forwards()
cmaxes = alphal.max(axis=1)
predictions = np.log(np.exp(alphal - cmaxes[:,None]).dot(s.trans_matrix)[:-1]) + cmaxes[:-1,None]
predictions = np.vstack((np.log(s.pi_0),predictions))
prediction_likes = np.logaddexp.reduce(predictions + s.aBl,axis=1) - np.concatenate(((0.,),np.logaddexp.reduce(alphal,axis=1)[:-1]))
computed_predictively = prediction_likes.sum()

print 'direct computation (backwards messages): %f' % computed_directly
print 'predictive computation (forwards HMM messages): %f' % computed_predictively

