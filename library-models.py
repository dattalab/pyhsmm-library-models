from __future__ import division
import numpy as np
na = np.newaxis

import pyhsmm
from pyhsmm.util.stats import sample_discrete_from_log_2d_destructive, getdatasize
import scipy.weave

### frozen mixture models, which will be the obs distributions for the library models

class FrozenLabels(pyhsmm.basic.pybasicbayes.internals.Labels):
    def __init__(self,likelihoods,*args,**kwargs):
        super(FrozenLabels,self).__init__(*args,**kwargs)
        self._likelihoods = likelihoods

    def meanfieldupdate(self):
        raise NotImplementedError

    def resample(self,temp=None):
        scores = self._likelihoods[self.data] + \
                self.weights.log_likelihood(np.arange(len(self.components)))

        if temp is not None:
            scores /= temp

        self.z = sample_discrete_from_log_2d_destructive(scores)

    def E_step(self):
        data, N, K = self.data, self.data.shape[0], len(self.components)
        self.expectations = np.empty((N,K))

        self.expectations = self._likelihoods[data]

        self.expectations += self.weights.log_likelihood(np.arange(K))

        self.expectations -= self.expectations.max(1)[:,na]
        np.exp(self.expectations,out=self.expectations)
        self.expectations /= self.expectations.sum(1)[:,na]

        self.z = self.expectations.argmax(1)

class FrozenMixtureDistribution(pyhsmm.basic.models.MixtureDistribution):
    def get_all_likelihoods(self,data):
        assert isinstance(data,np.ndarray)
        likelihoods = np.empty((data.shape[0],len(self.components)))
        for idx, c in enumerate(self.components):
            likelihoods[:,idx] = c.log_likelihood(data)
        maxes = likelihoods.max(axis=1)
        shifted_likelihoods = np.exp(likelihoods - maxes[:,na])
        return likelihoods, maxes, shifted_likelihoods

    def add_data(self,data,precomputed_likelihoods=None):
        raise NotImplementedError
        self.labels_list.append(FrozenLabels(
            data=np.asarray(data),
            components=self.components,
            weights=self.weights))

    def resample(self,likelihoods,niter=5,temp=None):
        if getdatasize(likelihoods) > 0:
            if isinstance(likelihoods,list):
                for l in likelihoods:
                    self.add_data(l)
            else:
                self.add_data(likelihoods)

            for itr in xrange(niter):
                self.resample_model(temp=temp)
            self.weights.resample([l.z for l in self.labels_list]) # for resampling concentration

            self._last_zs = [l.z for l in self.labels_list] # save the last labels we used

            self.labels_list = []
        else:
            self.resample_model(temp=temp)

    def resample_model(self, temp=None):
        for l in self.labels_list:
            l.resample(temp=temp)
        if hasattr(self.weights,'resample_just_weights'):
            self.weights.resample_just_weights([l.z for l in self.labels_list]) # don't do concentration
        else:
            self.weights.resample([l.z for l in self.labels_list])

    # TODO TODO x should be likelihoods now, not indices

    def _log_likelihoods(self,x):
        # NOTE: x must be indices
        return self.log_likelihood_faster_2(x)

    def log_likelihood_slower(self,x):
        # NOTE: x is indices
        K = len(self.components)
        vals = self._likelihoods[x.astype(np.int64)]
        vals += self.weights.log_likelihood(np.arange(K))
        return np.logaddexp.reduce(vals,axis=1)

    def log_likelihood_faster(self,sub_indices):
        # NOTE: this method takes INDICES into the data
        log_likelihoods = self._likelihoods
        log_weights = np.log(self.weights.weights)

        K = self.weights.weights.shape[0]
        num_sub_indices = sub_indices.shape[0]
        num_indices = log_likelihoods.shape[0]

        out = np.empty(num_sub_indices)

        scipy.weave.inline(
                '''
                using namespace Eigen;

                Map<ArrayXd> elog_weights(log_weights,K);
                Map<ArrayXXd> eall_log_likelihoods(log_likelihoods,K,num_indices);

                ArrayXd row(K);

                for (int i=0; i < num_sub_indices; i++) {
                    int idx = sub_indices[i];
                    row = elog_weights + eall_log_likelihoods.col(idx);
                    double themax = row.maxCoeff();
                    out[i] = log((row - themax).exp().sum()) + themax;
                }
                ''',['sub_indices','log_likelihoods','K','num_indices',
                    'num_sub_indices','out','log_weights'],
                headers=['<Eigen/Core>','<math.h>'],include_dirs=[pyhsmm.EIGEN_INCLUDE_DIR],
                extra_compile_args=['-O3','-DNDEBUG'])
        return out

    def log_likelihood_faster_2(self,sub_indices):
        # NOTE: this method takes INDICES into the data
        assert sub_indices.ndim == 1 and issubclass(sub_indices.dtype.type,np.integer)
        shifted_likelihoods = self._shifted_likelihoods
        maxes = self._maxes
        weights = self.weights.weights

        K = weights.shape[0]
        num_sub_indices = sub_indices.shape[0]
        num_indices = shifted_likelihoods.shape[0]

        out = np.empty(num_sub_indices)

        scipy.weave.inline(
                '''
                using namespace Eigen;

                Map<MatrixXd> eweights(weights,1,K);
                Map<MatrixXd> eall_likelihoods(shifted_likelihoods,K,num_indices);

                for (int i=0; i < num_sub_indices; i++) {
                    int idx = sub_indices[i];
                    out[i] = log((eweights * eall_likelihoods.col(idx)).array().value()) + maxes[idx];
                }
                ''',['sub_indices','shifted_likelihoods','K','num_indices',
                    'num_sub_indices','out','weights','maxes'],
                headers=['<Eigen/Core>','<math.h>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])
        assert not np.isnan(out).any()
        return out

    def max_likelihood(self,data,weights=None):
        # NOTE: data is an array or list of arrays of indices
        if weights is not None:
            raise NotImplementedError
        assert isinstance(data,list) or isinstance(data,np.ndarray)
        if isinstance(data,np.ndarray):
            data = [data]

        if getdatasize(data) > 0:
            for d in data:
                self.add_data(d)

            for itr in range(10):
                self.EM_step()

            for d in data:
                self.labels_list.pop()

    def EM_step(self):
        assert all(isinstance(c,MaxLikelihood) for c in self.components), \
                'Components must implement MaxLikelihood'
        assert len(self.labels_list) > 0, 'Must have data to run EM'

        ## E step
        for l in self.labels_list:
            l.E_step()

        ## M step
        # mixture weights
        self.weights.max_likelihood(np.arange(len(self.components)),
                [l.expectations for l in self.labels_list])

    def plot(self,data=[],color='b',label='',plot_params=True):
        # NOTE: this thing is weird; only plots the data in self._data, but
        # needs to be passed indices as data argument to this method. TODO
        if not isinstance(data,list):
            data = [data]
        for d in data:
            self.add_data(d)

        for l in self.labels_list:
            l.E_step() # sets l.z to MAP estimates
            for label, o in enumerate(self.components):
                if label in l.z:
                    o.plot(color=color,label=label,
                            data=self._data[l.data[l.z == label]] if l.data is not None else None)

        for d in data:
            self.labels_list.pop()

    def __getstate__(self):
        return dict(weights=self.weights)

    def __setstate__(self,d):
        self.weights = d['weights']
        # NOTE: need to set components library elsewhere!

### internals classes (states and labels)

class LibraryGMMLabels(pyhsmm.basic.pybasicbayes.internals.Labels):
    def resample(self,temp=None):
        shifted_likelihoods, maxes = \
                self.components[0]._shifted_likelihoods, self.components[0]._maxes
        allweights = np.hstack([c.weights.weights[:,na] for c in self.components])
        scores = np.log(shifted_likelihoods.dot(allweights))
        # scores += maxes[:,na] # this part isn't really necessary...

        if temp is not None:
            scores /= temp

        self.z = sample_discrete_from_log_2d_destructive(scores)

class LibraryHMMStates(pyhsmm.internals.states.HMMStatesEigen):
    def __init__(self,precomputed_likelihoods,data,*args,**kwargs):
        if precomputed_likelihoods is None:
            precomputed_likelihoods = self.obs_distns[0].get_all_likelihoods(data)
        self._likelihoods, self._shifted_likelihoods, self._maxes = precomputed_likelihoods
        super(LibraryHMMStates,self).__init__(*args,**kwargs)

    @property
    def aBl(self):
        # NOTE: we pretty much compute likelihoods here, both for efficiency and
        # because this class caches the data's likelihoods
        if self._aBl is None:
            shifted_likelihoods, maxes = self._shifted_likelihoods, self._maxes
            allweights = np.hstack([o.weights.weights[:,na] for o in self.obs_distns])
            scores = np.log(shifted_likelihoods.dot(allweights))
            scores += maxes[:,na]
            self._aBl = scores
        return self._aBl

class LibraryHSMMStatesIntegerNegativeBinomialVariant(pyhsmm.internals.statesHSMMStatesIntegerNegativeBinomialVariant,LibraryHMMStates):
    @property
    def hsmm_aBl(self):
        return LibraryHMMStates.aBl.fget(self)

### models

class LibraryGMM(pyhsmm.basic.models.Mixture):
    def add_data(self,data):
        self.labels_list.append(LibraryGMMLabels(data=np.asarray(data),
            components=self.components,weights=self.weights))

class LibraryHMM(pyhsmm.models.HMMEigen):
    _states_class = pyhsmm.internals.states.LibraryHMMStates

    def __init__(self,obs_distns,*args,**kwargs):
        assert all(isinstance(o,pyhsmm.basic.models.FrozenMixtureDistribution) for o in obs_distns) \
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

class LibraryHSMMIntNegBinVariant(pyhsmm.models.HSMMIntNegBinVariant,LibraryHMM):
    _states_class = pyhsmm.internals.states.LibraryHSMMStatesIntegerNegativeBinomialVariant

    def __init__(self,obs_distns,*args,**kwargs):
        assert all(isinstance(o,pyhsmm.basic.models.FrozenMixtureDistribution) for o in obs_distns) \
                and all(o.components is obs_distns[0].components for o in obs_distns)
        super(LibraryHSMMIntNegBinVariant,self).__init__(obs_distns,*args,**kwargs)

    def add_data(self,data,precomputed_likelihoods=None,**kwargs):
        LibraryHMM.add_data(self,data,precomputed_likelihoods,**kwargs)

    def resample_obs_distns(self,**kwargs):
        LibraryHMM.resample_obs_distns(self,**kwargs)

