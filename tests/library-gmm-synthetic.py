from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

from pyhsmm.basic.models import Mixture, MixtureDistribution
from library_models import FrozenMixtureDistribution, LibraryMM
from pyhsmm.basic.distributions import Gaussian, NegativeBinomialIntegerRVariantDuration
from pyhsmm.util.text import progprint_xrange

#############################
#  generate synthetic data  #
#############################

groups_in_metamm = 5
components_per_gmm = 2
component_hyperparameters = dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.01,nu_0=3)

GMMs = [MixtureDistribution(
    alpha_0=4.,
    components=[Gaussian(**component_hyperparameters) for i in range(components_per_gmm)])
    for state in range(groups_in_metamm)]

truemodel = Mixture(
        alpha_0=6,
        components=GMMs)

data, truelabels = truemodel.generate(2000)

#####################################
#  set up FrozenMixture components  #
#####################################

# list of all Gaussians
component_library = [c for m in GMMs for c in m.components]
library_size = len(component_library)

# initialize weights to indicator on one component
init_weights = np.eye(library_size)

meta_components = [FrozenMixtureDistribution(
    components=component_library,
    alpha_0=components_per_gmm,
    weights=row)
    for row in init_weights]

##############
#  build MM  #
##############

model = LibraryMM(
        alpha_0=6.,
        components=meta_components)

model.add_data(data)

##################
#  infer things  #
##################

for i in progprint_xrange(50):
    model.resample_model()

plt.figure()
truemodel.plot()
plt.gcf().suptitle('truth')

plt.figure()
model.plot()
plt.gcf().suptitle('inferred')

plt.show()
