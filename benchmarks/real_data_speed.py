from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

from pyhsmm.models import HSMMIntNegBinVariant
from pyhsmm.basic.models import MixtureDistribution
from library_models import FrozenMixtureDistribution, LibraryHSMMIntNegBinVariant
from pyhsmm.basic.distributions import Gaussian, NegativeBinomialIntegerRVariantDuration
from pyhsmm.util.text import progprint_xrange

from IPython.parallel import Client
Client()[:].execute(
'''
import __builtin__
__builtin__.__dict__['profile'] = lambda x: x
''')

###############
#  load data  #
###############

f = np.load("/home/alexbw/Data/C57-1-data-7-28-2013-fortesting.npz")
alldata = f['data']
means = f['means']
sigmas = f['sigmas']
labels = f['labels']

training_datas = np.array_split(alldata,2)

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
    alpha_0=30) for i in xrange(Nmax)]

################
#  build HSMM  #
################

dur_distns = [NegativeBinomialIntegerRVariantDuration(np.r_[0.,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1],alpha_0=5.,beta_0=5.)
        for state in range(Nmax)]

model = LibraryHSMMIntNegBinVariant(
        init_state_concentration=10.,
        alpha=6.,gamma=6.,
        obs_distns=obs_distns,
        dur_distns=dur_distns)

for data in training_datas:
    model.add_data_parallel(data,left_censoring=True)

##################
#  infer things  #
##################


for i in progprint_xrange(25):
    model.resample_model_parallel()

