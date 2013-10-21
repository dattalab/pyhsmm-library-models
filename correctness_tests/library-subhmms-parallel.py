from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm.util.text import progprint_xrange
from library_subhmm_models import HSMMIntNegBinVariantFrozenSubHMMs

from IPython.parallel import Client

PARALLEL = True

###############################
#  synthetic data generation  #
###############################

Nsuper = 5
Nsub = 10
T = 12000
obs_dim = 2

try:
    import brewer2mpl
    plt.set_cmap(brewer2mpl.get_map('Set1','qualitative',max(3,min(8,Nsuper))).mpl_colormap)
except:
    pass

obs_hypparams = dict(
        mu_0=np.zeros(obs_dim),
        sigma_0=np.eye(obs_dim),
        kappa_0=0.01,
        nu_0=obs_dim+10,
        )

dur_hypparams = dict(
        r_discrete_distn=np.r_[0,0,0,0,0,1.,1.,1.],
        alpha_0=20,
        beta_0=2,
        )

true_obs_distnss = [[pyhsmm.distributions.Gaussian(**obs_hypparams) for substate in xrange(Nsub)]
        for superstate in xrange(Nsuper)]

true_dur_distns = [pyhsmm.distributions.NegativeBinomialIntegerRVariantDuration(
    **dur_hypparams) for superstate in range(Nsuper)]

truemodel = pyhsmm.models.HSMMIntNegBinVariantSubHMMs(
        init_state_concentration=6,
        alpha=10.,gamma=10.,
        sub_alpha=10.,sub_gamma=10.,
        obs_distnss=true_obs_distnss,
        dur_distns=true_dur_distns)

datas = [truemodel.generate(T//6)[0] for i in range(6)]

plt.figure()
truemodel.plot()
plt.gcf().suptitle('truth')


##################
#  set up model  #
##################

Nmaxsuper = Nsuper*2

dur_distns = \
        [pyhsmm.distributions.NegativeBinomialIntegerRVariantDuration(
            **dur_hypparams) for superstate in range(Nmaxsuper)]

library = [o for obs_distns in true_obs_distnss for o in obs_distns]

model = HSMMIntNegBinVariantFrozenSubHMMs(
        init_state_concentration=6.,
        alpha=6.,gamma=6.,
        sub_alpha=6,sub_gamma=6,
        obs_distnss=[library]*Nmaxsuper,
        dur_distns=dur_distns)

if PARALLEL:
    for data in datas:
        model.add_data_parallel(data,left_censoring=True)
else:
    for data in datas:
        model.add_data(data,left_censoring=True)

###############
#  inference  #
###############

print 'starting resampling loop!'

if PARALLEL:
    for itr in progprint_xrange(50):
        model.resample_model_parallel()
else:
    for itr in progprint_xrange(50):
        model.resample_model()

plt.figure()
model.plot()
plt.gcf().suptitle('fit')

plt.show()

