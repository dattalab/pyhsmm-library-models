from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

from pyhsmm.models import HSMMIntNegBinVariant
from pyhsmm.basic.models import MixtureDistribution
from library_models import FrozenMixtureDistribution, LibraryHSMMIntNegBinVariant
from pyhsmm.basic.distributions import Gaussian, NegativeBinomialIntegerRVariantDuration
from pyhsmm.util.text import progprint_xrange

###############
#  load data  #
###############

f = np.load("/home/alexbw/Data/C57-1-data-7-28-2013-fortesting.npz")
alldata = f['data']
means = f['means']
sigmas = f['sigmas']

all_training_data = alldata[:60000]
training_datas = np.array_split(all_training_data,6)
test_data = alldata[-10000:]

#####################################
#  set up FrozenMixture components  #
#####################################

Nmax = 50

# list of all Gaussians
component_library = \
        [Gaussian(mu=mu,sigma=sigma,
            # hyperparameters not used
            mu_0=np.zeros_like(mu),sigma_0=np.eye(sigma.shape[0]),kappa_0=1.,nu_0=mu.shape[0]+5,
            ) for mu, sigma in zip(means,sigmas)]

library_size = len(component_library)

obs_distns = [FrozenMixtureDistribution(
    components=component_library,
    a_0=1.0,b_0=0.05)
    for i in xrange(Nmax)]

################
#  build HSMM  #
################

dur_distns = [NegativeBinomialIntegerRVariantDuration(np.r_[0.,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1],alpha_0=25.,beta_0=25.)
        for state in range(Nmax)]

model = LibraryHSMMIntNegBinVariant(
        init_state_concentration=10.,
        alpha_a_0=1.0,alpha_b_0=0.1,
        gamma_a_0=0.1,gamma_b_0=200.,
        obs_distns=obs_distns,
        dur_distns=dur_distns)

for data in training_datas:
    model.add_data_parallel(data,left_censoring=True)
    # model.add_data(data,left_censoring=True)

##################
#  infer things  #
##################


train_likes = []
for i in progprint_xrange(50):
    model.resample_model_parallel()
    # model.resample_model()
    train_likes.append(model.log_likelihood())

##########
#  plot  #
##########

plt.figure()
plt.plot(train_likes,label='training')
plt.legend()

plt.show()

