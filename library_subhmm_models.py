from __future__ import division
import numpy as np
from numpy import newaxis as na
import os, cPickle, hashlib

from pyhsmm.models import HSMMIntNegBinVariantSubHMMs
from pyhsmm.internals.states import HSMMIntNegBinVariantSubHMMsStates

from library_models import FrozenHMM

likelihood_cache_dir_subhmms = '/tmp/cached_likelihoods'

class HSMMIntNegBinVariantFrozenSubHMMsStates(HSMMIntNegBinVariantSubHMMsStates):
    def __init__(self,model,data,precomputed_likelihoods=None,**kwargs):
        self.model = model
        self.data = data
        if precomputed_likelihoods is None:
            self._frozen_aBls = self.get_all_likelihoods(model,data)
        super(HSMMIntNegBinVariantFrozenSubHMMsStates,self).__init__(model=model,data=data,**kwargs)

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
                frozen_aBls = cPickle.load(infile)
        else:
            self._aBls = None
            frozen_aBls = super(HSMMIntNegBinVariantFrozenSubHMMsStates,self).aBls
            with open(filepath,'w') as outfile:
                cPickle.dump(frozen_aBls,outfile,protocol=-1)
            print 'Computed and saved to cache: %s' % filename

        return frozen_aBls

    @property
    def aBls(self):
        return self._frozen_aBls

class HSMMIntNegBinVariantFrozenSubHMMs(HSMMIntNegBinVariantSubHMMs):
    _states_class = HSMMIntNegBinVariantFrozenSubHMMsStates
    _subhmm_class = FrozenHMM

