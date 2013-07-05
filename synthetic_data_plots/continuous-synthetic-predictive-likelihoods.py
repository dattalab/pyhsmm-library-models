from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import collections

import pyhsmm
from pyhsmm.util.text import progprint_xrange

resample_iter = 200

################
#  generation  #
################

N = 5

dur_hypparams = dict(r_discrete_distn=np.r_[0.,0.,0,0,0,0,0,0,0,0,1,1,1],alpha_0=1.,beta_0=10.)
true_dur_distns = [pyhsmm.distributions.NegativeBinomialIntegerRVariantDuration(**dur_hypparams)
        for state in range(N)]

obs_hypparams = {'mu_0':np.zeros(2),
                'sigma_0':np.eye(2),
                'kappa_0':0.05,
                'nu_0':2+2}
true_obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(N)]

true_trans_matrix = np.diag(np.repeat(0.5,N-1),k=1) + np.diag(np.repeat(0.5,N-1),k=-1)
true_trans_matrix /= true_trans_matrix.sum(1)[:,None]

truemodel = pyhsmm.models.HSMMIntNegBinVariant(alpha=10.,gamma=10.,init_state_concentration=6.,
                              obs_distns=true_obs_distns,
                              dur_distns=true_dur_distns)
truemodel.trans_distn.A = true_trans_matrix

training_datas = [truemodel.generate(500)[0] for i in range(10)]
test_data, _ = truemodel.generate(1000)

################
#  fit models  #
################

Nmax = 10
models = collections.OrderedDict()

### GMM

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
gmm = pyhsmm.basic.models.Mixture(alpha_0=N,components=obs_distns)

for training_data in training_datas:
    gmm.add_data(training_data)
for itr in progprint_xrange(resample_iter):
    gmm.resample_model()

models['GMM'] = gmm

### HMM

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
hmm = pyhsmm.models.StickyHMMEigen(alpha=N,gamma=N,init_state_concentration=10,kappa=30,
                obs_distns=obs_distns)

for training_data in training_datas:
    hmm.add_data(training_data)
for itr in progprint_xrange(resample_iter):
    hmm.resample_model()

models['HMM'] = hmm

### HSMM

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
dur_distns = [pyhsmm.distributions.NegativeBinomialIntegerRVariantDuration(**dur_hypparams)
                for state in xrange(Nmax)]
hsmm = pyhsmm.models.HSMMIntNegBinVariant(alpha=N,gamma=N,init_state_concentration=10,
                obs_distns=obs_distns,dur_distns=dur_distns)

for training_data in training_datas:
    hsmm.add_data(training_data)
for itr in progprint_xrange(resample_iter):
    hsmm.resample_model()

models['HSMM'] = hsmm

#########################
#  compare likelihoods  #
#########################

indices = range(1,500,1)

likelihoods = {name:m.predictive_likelihoods(test_data,indices) for name,m in models.iteritems()}

plt.figure()
for name, ls in likelihoods.iteritems():
    # plt.errorbar(indices,[np.mean(l) for l in ls],[l.std() for l in ls],label=name)
    plt.plot(indices,[np.mean(l) for l in ls],label=name)
plt.xlabel('time into future')
plt.ylabel('predictive likelihood')
plt.legend()

plt.show()

