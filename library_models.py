from __future__ import division
import numpy as np
na = np.newaxis
import copy, os, hashlib, cPickle
from warnings import warn
from collections import defaultdict

import pyhsmm
from pyhsmm.util.stats import sample_discrete_from_log_2d_destructive
from pyhsmm.util.general import engine_global_namespace

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

                scores_in = scores.copy() # TODO TODO HACK HACK REMOVE
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
    def __init__(self,data=None,precomputed_likelihoods=None,**kwargs):
        if data is not None:
            if precomputed_likelihoods is None:
                precomputed_likelihoods = kwargs['model'].obs_distns[0].get_all_likelihoods(data)
            self._likelihoods, self._shifted_likelihoods, self._maxes = precomputed_likelihoods
        super(LibraryHMMStates,self).__init__(data=data,**kwargs)

    @property
    def precomputed_likelihoods(self):
        return (self._likelihoods, self._shifted_likelihoods, self._maxes)

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

    def clear_caches(self):
        self._like = None
        super(LibraryHMMStates,self).clear_caches()

class LibraryHSMMStatesIntegerNegativeBinomialVariant(pyhsmm.internals.states.HSMMStatesIntegerNegativeBinomialVariant,LibraryHMMStates):
    def __init__(self,data=None,precomputed_likelihoods=None,**kwargs):
        if data is not None:
            if precomputed_likelihoods is None:
                precomputed_likelihoods = kwargs['model'].obs_distns[0].get_all_likelihoods(data)
            self._likelihoods, self._shifted_likelihoods, self._maxes = precomputed_likelihoods
        super(LibraryHSMMStatesIntegerNegativeBinomialVariant,self).__init__(data=data,**kwargs)

    @property
    def hsmm_aBl(self):
        return LibraryHMMStates.aBl.fget(self)

    def clear_caches(self):
        self._like = None
        super(LibraryHSMMStatesIntegerNegativeBinomialVariant,self).clear_caches()

class LibraryHSMMStatesINBVIndepTrans(LibraryHSMMStatesIntegerNegativeBinomialVariant):
    def __init__(self,model,group_id,**kwargs):
        self.group_id = group_id
        self._trans_distn = model.trans_distns[group_id]
        super(LibraryHSMMStatesINBVIndepTrans,self).__init__(model=model,**kwargs)

    @property
    def hsmm_trans_matrix(self):
        return self._trans_distn.A

    def clear_caches(self):
        self._like = None
        super(LibraryHSMMStatesINBVIndepTrans,self).clear_caches()

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
            if hasattr(self,'_last_resample_used_temp') and self._last_resample_used_temp:
                self._clear_caches()
            if all(hasattr(s,'_like') and s._like is not None for s in self.states_list):
                return sum(s._like for s in self.states_list)
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

    def truncate_num_states(self,target_num,destructive=False):
        # TODO this should just call add_data on new instead of dealing with
        # states
        # TODO states model pointer wasnt being set right...
        if not destructive:
            states_list = self.states_list
            self.states_list = []
            new = copy.deepcopy(self)
            self.states_list = states_list
        else:
            new = self

        # find most popular states
        counts = sum(np.bincount(s,minlength=len(self.obs_distns))
                for s in self.stateseqs_norep)
        most_popular = np.argsort(counts)[-target_num:]

        # limit trans distn, obs distns, dur distns, initial distn
        new.trans_distn.state_dim = target_num
        new.trans_distn.beta = new.trans_distn.beta[most_popular]
        new.trans_distn.beta /= new.trans_distn.beta.sum()
        new.trans_distn.A = new.trans_distn.A[np.ix_(most_popular,most_popular)]
        new.trans_distn.A /= new.trans_distn.A.sum(1)[:,na]
        new.trans_distn.fullA = new.trans_distn.fullA[np.ix_(most_popular,most_popular)]
        new.trans_distn.fullA /= new.trans_distn.fullA.sum(1)[:,na]

        new.obs_distns = [o for i,o in enumerate(new.obs_distns) if i in most_popular]

        new.dur_distns = [o for i,o in enumerate(new.dur_distns) if i in most_popular]

        new.init_state_distn.weights = new.init_state_distn.weights[most_popular]
        new.init_state_distn.weights /= new.init_state_distn.weights.sum()
        # NOTE: next line is just to be sure
        if hasattr(new,'left_censoring_init_state_distn'):
            self.left_censoring_init_state_distn._pi = None

        # set new state sequences to viterbi decodings given the new limited
        # parameters
        new.state_dim = target_num
        for s in new.states_list:
            s.model = new
            s.clear_caches()
            s.Viterbi()

        return new

    def remove_data_refs(self):
        for s in self.states_list:
            del s.data
            del s._likelihoods, s._shifted_likelihoods, s._maxes

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

    def add_data_parallel(self,data,**kwargs):
        import pyhsmm.parallel as parallel
        self.add_data(data=data,**kwargs)
        parallel.broadcast_data(self.states_list[-1].precomputed_likelihoods,
                costfunc=lambda x: x[0].shape[0])

    def resample_states_parallel(self,temp=None):
        import pyhsmm.parallel as parallel
        states = self.states_list
        self.states_list = [] # removed because we push the global model
        raw = parallel.map_on_each(
                self._state_sampler,
                [s.precomputed_likelihoods for s in states],
                kwargss=self._get_parallel_kwargss(states),
                engine_globals=dict(global_model=self,temp=temp),
                )
        self.states_list = states
        for s, (stateseq,like) in zip(self.states_list,raw):
            s.stateseq = stateseq
            s._like = like

    @staticmethod
    @engine_global_namespace # access to engine globals
    def _state_sampler(precomputed_likelihoods,**kwargs):
        # expects globals: global_model, temp
        assert len(global_model.states_list) == 0
        global_model.add_data(
                data=precomputed_likelihoods[0], # dummy
                precomputed_likelihoods=precomputed_likelihoods,
                initialize_from_prior=False,temp=temp,**kwargs)
        like = global_model.log_likelihood()
        return global_model.states_list.pop().stateseq, like

    def resample_obs_distns_parallel(self):
        import pyhsmm.parallel as parallel
        first_call = hasattr(self,'_called_resample_obs_distns_parallel')
        # NOTE: don't really need to transmit weights
        raw = parallel.call_with_all(
                self._obs_sampler,
                [s.precomputed_likelihoods for s in self.states_list],
                kwargss=[dict(weights=o.weights.weights,
                      indicess=[s.stateseq == i for s in self.states_list])
                    for i,o in enumerate(self.obs_distns)],
                engine_globals=dict(obs_distn_template=self.obs_distns[0])
                    if not first_call else None
                )
        self._called_resample_obs_distns_parallel = True
        for o,weights in zip(self.obs_distns,raw):
            o.weights.weights = weights

    @staticmethod
    @engine_global_namespace # access to engine globals
    def _obs_sampler(precomputed_likelihoodss,weights,indicess):
        obs_distn_template.weights.weights = weights # don't really need this
        obs_distn_template.resample_from_likelihoods([precomputed_likelihoods[0][indices]
            for precomputed_likelihoods, indices in zip(precomputed_likelihoodss,indicess)])
        return obs_distn_template.weights.weights

class LibraryHSMMIntNegBinVariant(LibraryHMM,pyhsmm.models.HSMMIntNegBinVariant):
    _states_class = LibraryHSMMStatesIntegerNegativeBinomialVariant

    def __init__(self,obs_distns,*args,**kwargs):
        assert all(isinstance(o,FrozenMixtureDistribution) for o in obs_distns) \
                and all(o.components is obs_distns[0].components for o in obs_distns)
        pyhsmm.models.HSMMIntNegBinVariant.__init__(self,obs_distns=obs_distns,*args,**kwargs)

    def _clear_caches(self):
        LibraryHMM._clear_caches(self)
        pyhsmm.models.HSMMIntNegBinVariant._clear_caches(self)

    def unfreeze(self,destructive=False):
        if destructive:
            obs_distns = [MixtureDistribution(weights=o.weights,components=o.components)
                for o in self.obs_distns]
            dur_distns = self.dur_distns
            trans_distn = self.trans_distn
        else:
            obs_distns = [MixtureDistribution(
                    weights=o.weights,
                    components=[copy.deepcopy(c) for c in o.components])
                for o in self.obs_distns]
            dur_distns = [copy.deepcopy(d) for d in self.dur_distns]

        raise NotImplementedError

        # TODO init state distn? 

    def log_likelihood(self,data=None,precomputed_likelihoods=None,**kwargs):
        if data is not None:
            self.add_data(data=data,precomputed_likelihoods=precomputed_likelihoods,**kwargs)
            s = self.states_list.pop()
            betal,superbetal = s.messages_backwards()
            return np.logaddexp.reduce(np.log(s.pi_0) + betal[0] + s.aBl[0])
        else:
            if hasattr(self,'_last_resample_used_temp') and self._last_resample_used_temp:
                self._clear_caches()
            if all(hasattr(s,'_like') and s._like is not None for s in self.states_list):
                return sum(s._like for s in self.states_list)
            else:
                return super(LibraryHSMMIntNegBinVariant,self).log_likelihood()

    def Viterbi_EM_step(self):
        super(LibraryHSMMIntNegBinVariant,self).Viterbi_EM_step()

        # M step for duration distributions
        for state, distn in enumerate(self.dur_distns):
            distn.max_likelihood(
                    [s.durations[s.stateseq_norep == state] for s in self.states_list])

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

    def __init__(self,*args,**kwargs):
        super(LibraryHSMMIntNegBinVariantIndepTrans,self).__init__(*args,**kwargs)
        # self.trans_distn is a template object for all the trans distns
        self.trans_distns = defaultdict(lambda: copy.deepcopy(self.trans_distn))

    def resample_trans_distn(self):
        for group_id, trans_distn in self.trans_distns.iteritems():
            trans_distn.resample([s.stateseq for s in self.states_list
                if s.group_id == group_id])
        self._clear_caches()

    def _get_parallel_kwargss(self,states_objs):
        outs = super(LibraryHSMMStatesINBVIndepTrans,self)._get_parallel_kwargss(states_objs)
        return [dict(group_id=s.group_id,**out) for s,out in zip(states_objs,outs)]

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

