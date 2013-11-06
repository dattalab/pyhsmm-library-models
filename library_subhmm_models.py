from __future__ import division
import numpy as np
from numpy import newaxis as na
import os, cPickle, hashlib, tempfile

import pyhsmm
from pyhsmm.models import HSMMIntNegBinVariantSubHMMs
from pyhsmm.internals.states import HSMMIntNegBinVariantSubHMMsStates
from pyhsmm.util.general import engine_global_namespace

import socket
hostname = socket.gethostname()
if os.path.exists("/hms/scratch1/"):
    likelihood_cache_dir_subhmms = '/hms/scratch1/abw11/tmp/cached_likelihoods'
    likelihood_cache_dir_hmm = '/hms/scratch1/abw11/tmp/cached_likelihoods_hmm'
else:
    tempdir = tempfile.gettempdir()
    likelihood_cache_dir_subhmms = os.path.join(tempdir, 'cached_likelihoods')
    likelihood_cache_dir_hmm = os.path.join(tempdir, 'cached_likelihoods_hmm')

class FrozenSubHMM(pyhsmm.models.HMMEigen):
    def resample_obs_distns(self,*args,**kwargs):
        pass

    def resample_states(self,*args,**kwargs):
        pass


class HSMMIntNegBinVariantFrozenSubHMMsStates(HSMMIntNegBinVariantSubHMMsStates):
    # NOTE: assumes all subHMMs are the same, so that all frozen_aBls are same
    def __init__(self,model,data,frozen_aBl=None,**kwargs):
        self.model = model
        self.data = data
        if frozen_aBl is None:
            self._frozen_aBls = self.get_all_likelihoods(model,data)
        else:
            self._frozen_aBls = [frozen_aBl] * self.hsmm_trans_matrix.shape[0]
        super(HSMMIntNegBinVariantFrozenSubHMMsStates,self).__init__(
                model=model,data=data,**kwargs)

    # TODO compute likelihoods lazily? push this into aBls? why'd I break it
    # out? then I'd need to push data... need lazy loading too
    def get_all_likelihoods(self,model,data):
        if not os.path.isdir(likelihood_cache_dir_subhmms):
            os.mkdir(likelihood_cache_dir_subhmms)
        thehash = hashlib.sha1(data)

        # NOTE: assumes model.obs_distnss is a list of repeated obs_distns lists

        for o in model.obs_distnss[0]:
            thehash.update(o.mu)
            thehash.update(o.sigma)
        filename = thehash.hexdigest()
        filepath = os.path.join(likelihood_cache_dir_subhmms,filename)

        if os.path.isfile(filepath):
            with open(filepath,'r') as infile:
                frozen_aBl = cPickle.load(infile)
        else:
            self._aBls = None
            frozen_aBl = super(HSMMIntNegBinVariantFrozenSubHMMsStates,self).aBls[0]
            with open(filepath,'w') as outfile:
                cPickle.dump(frozen_aBl,outfile,protocol=-1)
            print 'Computed and saved to cache: %s' % filename

        return [frozen_aBl] * self.hsmm_trans_matrix.shape[0] \
                if isinstance(frozen_aBl,np.ndarray) else frozen_aBl

    @property
    def aBls(self):
        return self._frozen_aBls

class HSMMIntNegBinVariantFrozenSubHMMs(HSMMIntNegBinVariantSubHMMs):
    _states_class = HSMMIntNegBinVariantFrozenSubHMMsStates
    _subhmm_class = FrozenSubHMM

    def add_data_parallel(self,data,**kwargs):
        import pyhsmm.parallel as parallel
        self.add_data(data=data,**kwargs)
        parallel.add_data(self.states_list[-1]._frozen_aBls[0])

    # TODO is this necessary, or is it inherited?
    def _get_parallel_kwargss(self,states_objs):
        return [dict(trunc=s.trunc,left_censoring=s.left_censoring,
                    right_censoring=s.right_censoring) for s in states_objs]

    def resample_states_parallel(self,states_to_resample,states_to_hold_out,temp=None):
        import pyhsmm.parallel as parallel
        # remove things we don't need in parallel
        self.states_list = []
        # NOTE: we could also call s._remove_substates_from_subHMMs, but we
        # don't want to send ANY substates objects, so we remove everything here
        # and add the un-resampled substates objects back to the HMMs at the end
        # of this method
        for s in states_to_resample:
            s.substates_list = []
        for hmm in self.HMMs:
            hmm.states_list = [] # includes held out subseqs, added back below

        raw = parallel.map_on_each(
                self._state_sampler,
                [s._frozen_aBls[0] for s in states_to_resample],
                kwargss=self._get_parallel_kwargss(states_to_resample),
                engine_globals=dict(global_model=self,temp=temp))

        for s, (big_stateseq,like) in zip(states_to_resample,raw):
            s.big_stateseq = big_stateseq
            s._map_states() # NOTE: calls s._add_substates_to_subhmms
            s._loglike = like

        # these got cleared above, so we add them back
        for s in states_to_hold_out:
            s._add_substates_to_subHMMs()

        return states_to_resample

    @staticmethod
    @engine_global_namespace
    def _state_sampler(frozen_aBl,**kwargs):
        # expects globals: global_model, temp
        # NOTE: a little extra work because this runs _map_states locally
        # TODO allocate alphan out here (global)
        global_model.add_data(
                data=frozen_aBl, # dummy
                frozen_aBl=frozen_aBl,
                initialize_from_prior=False,temp=temp,**kwargs)
        like = global_model.states_list[-1].log_likelihood()
        big_stateseq = global_model.states_list.pop().big_stateseq
        return big_stateseq, like

