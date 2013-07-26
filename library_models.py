from __future__ import division
import numpy as np
na = np.newaxis
import copy, os, hashlib, cPickle
from warnings import warn

import pyhsmm
from pyhsmm.util.stats import sample_discrete_from_log_2d_destructive
from pyhsmm.util.general import interactive

### frozen mixture distributions, which will be the obs distributions for the library models

# likelihood_cache_dir = os.path.join(os.path.dirname(__file__), 'cached_likelihoods')
likelihood_cache_dir = '/tmp/cached_likelihoods'

class FrozenMixtureDistribution(pyhsmm.basic.models.MixtureDistribution):
    def get_all_likelihoods(self,data):
        # NOTE: doesn't reference self.weights; it's just against
        # self.components. this method is for the model to call inside add_data

        if not os.path.isdir(likelihood_cache_dir):
            os.mkdir(likelihood_cache_dir)

        thehash = hashlib.sha1(data)
        for c in self.components:
            thehash.update(c.mu)
            thehash.update(c.sigma)
        filename = thehash.hexdigest()
        filepath = os.path.join(likelihood_cache_dir,filename)

        if os.path.isfile(filepath):
            with open(filepath,'r') as infile:
                likelihoods, shifted_likelihoods, maxes = cPickle.load(infile)
            # print 'Loaded from cache: %s' % filename
        else:
            likelihoods = np.empty((data.shape[0],len(self.components)))
            for idx, c in enumerate(self.components):
                likelihoods[:,idx] = c.log_likelihood(data)
            maxes = likelihoods.max(axis=1)
            shifted_likelihoods = np.exp(likelihoods - maxes[:,na])

            with open(filepath,'w') as outfile:
                cPickle.dump((likelihoods,shifted_likelihoods,maxes),outfile,protocol=-1)

            print 'Computed and saved to cache: %s' % filename

        return likelihoods, shifted_likelihoods, maxes

    def log_likelihood(self,data_likelihoods):
        warn('dont call me, call shifted version for speed!')
        vals = data_likelihoods + self.weights.weights
        return np.logaddexp.reduce(vals,axis=1)

    def log_likelihoods_shifted(self,shifted_likelihoods,maxes):
        return np.log(shifted_likelihoods.dot(self.weights.weights)) + maxes

    def resample(self,*args,**kwargs):
        raise NotImplementedError, 'should be calling resample_from_likelihoods instead'

    def resample_from_likelihoods(self,data_likelihoods=[],niter=5,temp=None):
        if isinstance(data_likelihoods,list) and len(data_likelihoods) > 0:
            data_likelihoods = np.concatenate(data_likelihoods)

        if len(data_likelihoods) > 0:
            for itr in xrange(niter):
                scores = data_likelihoods + self.weights.weights

                if temp is not None:
                    scores /= temp

                scores = np.clip(scores, -1e200, 1e200) # TODO: HACK HACK HACK
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
    def __init__(self,data,precomputed_likelihoods=None,**kwargs):
        if precomputed_likelihoods is None:
            precomputed_likelihoods = kwargs['model'].obs_distns[0].get_all_likelihoods(data)
        self._likelihoods, self._shifted_likelihoods, self._maxes = precomputed_likelihoods
        super(LibraryHMMStates,self).__init__(data=data,**kwargs)

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
    def __init__(self,data,precomputed_likelihoods=None,**kwargs):
        if precomputed_likelihoods is None:
            precomputed_likelihoods = kwargs['model'].obs_distns[0].get_all_likelihoods(data)
        self._likelihoods, self._shifted_likelihoods, self._maxes = precomputed_likelihoods
        super(LibraryHSMMStatesIntegerNegativeBinomialVariant,self).__init__(data=data,**kwargs)

    @property
    def hsmm_aBl(self):
        return LibraryHMMStates.aBl.fget(self)

class LibraryHSMMStatesINBVIndepTrans(LibraryHSMMStatesIntegerNegativeBinomialVariant):
    def __init__(self,model,**kwargs):
        self._trans_distn = copy.deepcopy(model.trans_distn)
        self._trans_distn.resample()
        super(LibraryHSMMStatesINBVIndepTrans,self).__init__(model=model,**kwargs)

    @property
    def hsmm_trans_matrix(self):
        return self._trans_distn.A

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

    # this method is necessary because mixture models handle likelihood
    # calculation differently from the HMMs (less work in labels object, more
    # work here, HMMs make a temporary labels object)
    def _log_likelihoods(self,x):
        _, shifted_likelihoods, maxes = self.components[0].get_all_likelihoods(x)
        vals = np.empty((x.shape[0],len(self.components)))
        for idx, c in enumerate(self.components):
            vals[:,idx] = c.log_likelihoods_shifted(shifted_likelihoods,maxes)
        vals += np.log(self.weights.weights)
        return np.logaddexp.reduce(vals,axis=1)

    def resample_model(self,**kwargs):
        for idx, c in enumerate(self.components):
            c.resample_from_likelihoods(
                    [l._likelihoods[l.z == idx] for l in self.labels_list],
                    **kwargs)

        self.weights.resample([l.z for l in self.labels_list])

        for l in self.labels_list:
            l.resample(**kwargs)

    def Viterbi_EM_step(self):
        for l in self.labels_list:
            l.E_step()

        for idx, c in enumerate(self.components):
            c.max_likelihood_from_likelihoods(
                    [l._likelihoods[l.z == idx] for l in self.labels_list])

        self.weights.max_likelihood([l.z for l in self.labels_list])

    def EM_step(self):
        raise NotImplementedError

    def remove_data_refs(self):
        for l in self.labels_list:
            del l.data

    def reset(self,labels=None,list_of_labels=None):
        if labels is not None:
            assert len(self.labels_list) == 1
            list_of_labels = [labels]
        assert list_of_labels is None or len(list_of_labels) == len(self.labels_list)

        for c in self.components:
            c.resample_from_likelihoods()

        self.weights.resample()

        if list_of_labels is None:
            for l in self.labels_list:
                l._generate(len(l.z))
        else:
            for l,z in zip(self.labels,list_of_labels):
                l.z = z

class LibraryHMM(pyhsmm.models.HMMEigen):
    _states_class = LibraryHMMStates

    def __init__(self,obs_distns,*args,**kwargs):
        assert all(isinstance(o,FrozenMixtureDistribution) for o in obs_distns) \
                and all(o.components is obs_distns[0].components for o in obs_distns)
        super(LibraryHMM,self).__init__(obs_distns,*args,**kwargs)

    def log_likelihood(self,data=None,precomputed_likelihoods=None):
        if data is not None:
            self.add_data(data=data,precomputed_likelihoods=precomputed_likelihoods)
            s = self.states_list.pop()
            betal = s.messages_backwards()
            return np.logaddexp.reduce(np.log(s.pi_0) + betal[0] + s.aBl[0])
        else:
            return super(LibraryHMM,self).log_likelihood()

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

    def remove_data_refs(self):
        for s in self.states_list:
            del s.data

    def reset(self,stateseq=None,list_of_stateseqs=None):
        if stateseq is not None:
            list_of_stateseqs = [stateseq]
        assert list_of_stateseqs is None or len(list_of_stateseqs) == len(self.states_list)

        for o in self.obs_distns:
            o.resample_from_likelihoods()

        self.trans_distn.resample()

        if list_of_stateseqs is None:
            for s in self.states_list:
                s.generate_states(len(s.data))
        else:
            for s,stateseq in zip(self.states_list,list_of_stateseqs):
                s.stateseq = stateseq

class LibraryHSMMIntNegBinVariant(LibraryHMM,pyhsmm.models.HSMMIntNegBinVariant):
    _states_class = LibraryHSMMStatesIntegerNegativeBinomialVariant

    def __init__(self,obs_distns,*args,**kwargs):
        assert all(isinstance(o,FrozenMixtureDistribution) for o in obs_distns) \
                and all(o.components is obs_distns[0].components for o in obs_distns)
        pyhsmm.models.HSMMIntNegBinVariant.__init__(self,obs_distns,*args,**kwargs)

    def log_likelihood(self,data=None,precomputed_likelihoods=None,**kwargs):
        if data is not None:
            self.add_data(data=data,precomputed_likelihoods=precomputed_likelihoods,**kwargs)
            s = self.states_list.pop()
            betal,superbetal = s.messages_backwards()
            return np.logaddexp.reduce(np.log(s.pi_0) + betal[0] + s.aBl[0])
        else:
            # import pudb
            # pudb.set_trace()
            return super(LibraryHSMMIntNegBinVariant,self).log_likelihood()

    def Viterbi_EM_step(self):
        super(LibraryHSMMIntNegBinVariant,self).Viterbi_EM_step()

        # M step for duration distributions
        for state, distn in enumerate(self.dur_distns):
            distn.max_likelihood(
                    [s.durations[s.stateseq_norep == state] for s in self.states_list])

    def add_data_parallel(self,data_id,**kwargs):
        import parallel # not pyhsmm.parallel
        self.add_data(data=parallel.alldata[data_id],
                precomputed_likelihoods=parallel.alllikelihoods[data_id],
                **kwargs)
        self.states_list[-1].data_id = data_id

    def _build_states_parallel(self,states_to_resample,temp=None):
        import parallel # not pyhsmm.parallel
        parallel.dv.push(dict(temp=temp),block=False)
        raw_stateseq_tuples = parallel.dv.map(self._state_builder,
                [(s.data_id,s.left_censoring) for s in states_to_resample],block=True)

        for data_id, left_censoring, stateseq, stateseq_norep, durations in raw_stateseq_tuples:
            self.add_data(
                    data=parallel.alldata[data_id],
                    precomputed_likelihoods=parallel.alllikelihoods[data_id],
                    stateseq=stateseq,
                    stateseq_norep=stateseq_norep,
                    durations=durations,
                    left_censoring=left_censoring)
            self.states_list[-1].data_id = data_id

    @staticmethod
    @interactive
    def _state_builder((data_id,left_censoring)):
        # expects globals: global_model, alldata, alllikelihoods, temp
        global_model.add_data(
                data=alldata[data_id],
                precomputed_likelihoods=alllikelihoods[data_id],
                initialize_from_prior=False,
                left_censoring=left_censoring,
                temp=temp)
        s = global_model.states_list.pop()
        stateseq, stateseq_norep, durations = s.stateseq, s.stateseq_norep, s.durations

        return (data_id, left_censoring, stateseq, stateseq_norep, durations)

    def reset(self,stateseq_norep=None,durations=None,
            list_of_stateseq_noreps=None,list_of_durations=None):
        if stateseq_norep is not None:
            list_of_stateseq_noreps = [stateseq_norep]
            list_of_durations = [durations]
        assert list_of_stateseq_noreps is None or len(self.states_list) == len(list_of_stateseq_noreps) == len(list_of_durations)

        for o in self.obs_distns:
            o.resample_from_likelihoods()

        self.trans_distn.resample()

        if list_of_stateseq_noreps is None:
            for s in self.states_list:
                s.generate_states()
        else:
            for s,stateseq_norep,durations in zip(self.states_list,
                    list_of_stateseq_noreps,list_of_durations):
                s.stateseq_norep = stateseq_norep
                s.durations = durations
                s.stateseq = np.asarray(stateseq_norep).repeat(durations)[:len(s.data)]

class LibraryHSMMIntNegBinVariantIndepTrans(LibraryHSMMIntNegBinVariant):
    _states_class = LibraryHSMMStatesINBVIndepTrans

    def resample_trans_distn(self):
        for s in self.states_list:
            s._trans_distn.resample([s.stateseq])
        self._clear_caches()

### models that fix the syllables

class LibraryMMFixedObs(LibraryMM):
    def resample_model(self,**kwargs):
        self.weights.resample([l.z for l in self.labels_list])

        for l in self.labels_list:
            l.resample(**kwargs)

class LibraryHMMFixedObs(LibraryHMM):
    def resample_obs_distns(self,*args,**kwargs):
        pass

class LibraryHSMMIntNegBinVariantFixedObs(LibraryHMMFixedObs,LibraryHSMMIntNegBinVariant):
    pass

