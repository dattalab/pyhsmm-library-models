from __future__ import division
import numpy as np
na = np.newaxis
import copy
from warnings import warn

import pyhsmm
from pyhsmm.util.stats import sample_discrete_from_log_2d_destructive

### frozen mixture distributions, which will be the obs distributions for the library models

class FrozenMixtureDistribution(pyhsmm.basic.models.MixtureDistribution):
    def get_all_likelihoods(self,data):
        # NOTE: doesn't reference self.weights; it's just against
        # self.components. this method is for the model to call inside add_data
        likelihoods = np.empty((data.shape[0],len(self.components)))
        for idx, c in enumerate(self.components):
            likelihoods[:,idx] = c.log_likelihood(data)
        maxes = likelihoods.max(axis=1)
        shifted_likelihoods = np.exp(likelihoods - maxes[:,na])
        return likelihoods, shifted_likelihoods, maxes

    def log_likelihood(self,data_likelihoods):
        warn('dont call me, call shifted version for speed!')
        vals = data_likelihoods + self.weights.weights
        return np.logaddexp.reduce(vals,axis=1)

    def log_likelihoods_shifted(self,shifted_likelihoods,maxes):
        return np.log(shifted_likelihoods.dot(self.weights.weights)) + maxes

    def resample(self,*args,**kwargs):
        raise NotImplementedError, 'should be calling resample_from_likelihoods instead'

    def resample_from_likelihoods(self,data_likelihoods,niter=5,temp=None):
        if isinstance(data_likelihoods,list):
            data_likelihoods = np.concatenate(data_likelihoods)

        if data_likelihoods.shape[0] > 0:
            for itr in xrange(niter):
                scores = data_likelihoods + self.weights.weights

                if temp is not None:
                    scores /= temp

                z = sample_discrete_from_log_2d_destructive(scores)

                if hasattr(self.weights,'resample_just_weights'):
                    self.weights.resample_just_weights(z)
                else:
                    self.weights.resample(z)

            self.weights.resample(z) # for concentration parameter

        else:
            self.weights.resample()

    def max_likelihood(self,*args,**kwargs):
        raise NotImplementedError, 'should be calling max_likelihood_from_likelihoods instead'

    def max_likelihood_from_likelihoods(self,data_likelihoods,niter=10):
        if isinstance(data_likelihoods,list):
            data_likelihoods = np.concatenate(data_likelihoods)

        if data_likelihoods.shape[0] > 0:
            for itr in xrange(niter):
                ## E step
                scores = data_likelihoods + self.weights.weights

                ## M step
                scores -= scores.max(1)[:,na]
                np.exp(scores,out=scores)
                scores /= scores.sum(1)[:,na]
                self.weights.weights = scores.sum(0)
                self.weights.weights /= self.weights.weights.sum()

    def copy_sample(self):
        new = copy.copy(self)
        new.weights = self.weights.copy_sample()
        return new

### internals classes (states and labels)

class LibraryMMLabels(pyhsmm.basic.pybasicbayes.internals.labels.Labels):
    def __init__(self,precomputed_likelihoods,data,**kwargs):
        super(LibraryMMLabels,self).__init__(data=data,**kwargs)
        if precomputed_likelihoods is None:
            precomputed_likelihoods = self.components[0].get_all_likelihoods(data)
        self._likelihoods, self._shifted_likelihoods, self._maxes = precomputed_likelihoods

    def resample(self,temp=None):
        shifted_likelihoods, maxes = \
                self._shifted_likelihoods, self._maxes
        allweights = np.hstack([c.weights.weights[:,na] for c in self.components])
        scores = np.log(shifted_likelihoods.dot(allweights)) + np.log(self.weights.weights)

        if temp is not None:
            scores /= temp

        self.z = sample_discrete_from_log_2d_destructive(scores)

class LibraryHMMStates(pyhsmm.internals.states.HMMStatesEigen):
    def __init__(self,precomputed_likelihoods,data,**kwargs):
        super(LibraryHMMStates,self).__init__(data=data,**kwargs)
        if precomputed_likelihoods is None:
            precomputed_likelihoods = self.obs_distns[0].get_all_likelihoods(data)
        self._likelihoods, self._shifted_likelihoods, self._maxes = precomputed_likelihoods

    @property
    def aBl(self):
        # we use dot to compute all the likelihoods here for efficiency
        if self._aBl is None:
            shifted_likelihoods, maxes = self._shifted_likelihoods, self._maxes
            allweights = np.hstack([o.weights.weights[:,na] for o in self.obs_distns])
            scores = np.log(shifted_likelihoods.dot(allweights))
            scores += maxes[:,na]
            self._aBl = np.nan_to_num(scores)
        return self._aBl

class LibraryHSMMStatesIntegerNegativeBinomialVariant(pyhsmm.internals.states.HSMMStatesIntegerNegativeBinomialVariant,LibraryHMMStates):
    def __init__(self,precomputed_likelihoods,data,**kwargs):
        super(LibraryHSMMStatesIntegerNegativeBinomialVariant,self).__init__(data=data,**kwargs)
        if precomputed_likelihoods is None:
            precomputed_likelihoods = self.obs_distns[0].get_all_likelihoods(data)
        self._likelihoods, self._shifted_likelihoods, self._maxes = precomputed_likelihoods

    @property
    def hsmm_aBl(self):
        return LibraryHMMStates.aBl.fget(self)

### models

class LibraryMM(pyhsmm.basic.models.Mixture):
    def __init__(self,components,*args,**kwargs):
        assert all(isinstance(o,FrozenMixtureDistribution) for o in components) \
                and all(o.components is components[0].components for o in components)
        super(LibraryMM,self).__init__(components,*args,**kwargs)

    def add_data(self,data,precomputed_likelihoods=None,**kwargs):
        self.labels_list.append(LibraryMMLabels(data=np.asarray(data),
            components=self.components,weights=self.weights,
            precomputed_likelihoods=precomputed_likelihoods))

    def resample_model(self,temp=None):
        for l in self.labels_list:
            l.resample(temp=temp)

        for idx, c in enumerate(self.components):
            c.resample_from_likelihoods(
                    [l._likelihoods[l.z == idx] for l in self.labels_list],
                    temp=temp)

        self.weights.resample([l.z for l in self.labels_list])

    def Viterbi_EM_step(self):
        for l in self.labels_list:
            l.E_step()

        for idx, c in enumerate(self.components):
            c.max_likelihood_from_likelihoods(
                    [l._likelihoods[l.z == idx] for l in self.labels_list])

        self.weights.max_likelihood([l.z for l in self.labels_list])

    def EM_step(self):
        raise NotImplementedError


class LibraryHMM(pyhsmm.models.HMMEigen):
    _states_class = LibraryHMMStates

    def __init__(self,obs_distns,*args,**kwargs):
        assert all(isinstance(o,FrozenMixtureDistribution) for o in obs_distns) \
                and all(o.components is obs_distns[0].components for o in obs_distns)
        super(LibraryHMM,self).__init__(obs_distns,*args,**kwargs)

    def add_data(self,data,precomputed_likelihoods=None,**kwargs):
        self.states_list.append(self._states_class(model=self,data=np.asarray(data),
            precomputed_likelihoods=precomputed_likelihoods,**kwargs))

    def resample_obs_distns(self,**kwargs):
        for state, distn in enumerate(self.obs_distns):
            distn.resample_from_likelihoods(
                    [s._likelihoods[s.stateseq == state] for s in self.states_list],
                    **kwargs)
        self._clear_caches()

    def Viterbi_EM_step(self):
        # NOTE: mostly same as parent, except we call
        # max_likelihood_from_likelihoods and pass it s._likelihoods

        assert len(self.states_list) > 0, 'Must have data to run Viterbi EM'
        self._clear_caches()

        ## Viterbi step
        for s in self.states_list:
            s.Viterbi()

        ## M step
        # observation distribution parameters
        for state, distn in enumerate(self.obs_distns):
            distn.max_likelihood_from_likelihoods(
                    [s._likelihoods[s.stateseq == state] for s in self.states_list])

        # initial distribution parameters
        self.init_state_distn.max_likelihood(
                np.array([s.stateseq[0] for s in self.states_list]))

        # transition parameters (requiring more than just the marginal expectations)
        self.trans_distn.max_likelihood([s.stateseq for s in self.states_list])


class LibraryHSMMIntNegBinVariant(LibraryHMM,pyhsmm.models.HSMMIntNegBinVariant):
    _states_class = LibraryHSMMStatesIntegerNegativeBinomialVariant

    def __init__(self,obs_distns,*args,**kwargs):
        assert all(isinstance(o,FrozenMixtureDistribution) for o in obs_distns) \
                and all(o.components is obs_distns[0].components for o in obs_distns)
        pyhsmm.models.HSMMIntNegBinVariant.__init__(self,obs_distns,*args,**kwargs)

    def Viterbi_EM_step(self):
        super(LibraryHSMMIntNegBinVariant,self).Viterbi_EM_step()

        # M step for duration distributions
        for state, distn in enumerate(self.dur_distns):
            distn.max_likelihood(
                    [s.durations[s.stateseq_norep == state] for s in self.states_list])

