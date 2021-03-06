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

train_likes = []
test_likes = []

for i in progprint_xrange(50):
    model.resample_model()
    # model.resample_model_parallel()
    train_likes.append(model.log_likelihood())
    # test_likes.append(model.log_likelihood(test_data,left_censoring=True))

print 'training data likelihood when in the model: %g' % model.log_likelihood()
print 'training data likelihood passed in externally: %g' % sum(model.log_likelihood(data,left_censoring=True) for data in training_datas)

plt.figure()
truemodel.plot()
plt.gcf().suptitle('truth')

plt.figure()
model.plot()
plt.gcf().suptitle('inferred')

# plt.figure()
# plt.plot(train_likes,label='training')
# plt.plot(test_likes,label='test')
# plt.legend()

plt.show()

