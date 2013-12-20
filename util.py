from __future__ import division
import numpy as np
from collections import defaultdict
from itertools import count
import os, hashlib, tempfile, cPickle, socket

# IDEA: restrict to top by munging trans matrix, then viterbi. with copy! or an
# option to be destructive

def reduce_to_most_used(stateseq,target_num_states):
    most_freq = np.argsort(np.bincount(stateseq,minlength=target_num_states))[-target_num_states:]

    freq_map = defaultdict(count().next)
    rare_map = defaultdict(lambda: np.random.randint(target_num_states))

    out_stateseq = np.empty_like(stateseq)
    for i,s in enumerate(stateseq):
        out_stateseq[i] = freq_map[s] if s in most_freq else rare_map[s]
    return out_stateseq


def reduce_random(stateseq,target_num_states):
    themap = defaultdict(lambda: np.random.randint(target_num_states))
    out_stateseq = np.empty_like(stateseq)
    for i,s in enumerate(stateseq):
        out_stateseq[i] = themap[s]
    return out_stateseq


### likelihood caching
if os.path.exists("/data/behavior/"):
    tempdir = "/data/behavior/"
else:
    tempdir = tempfile.gettempdir()
full_likelihood_caches = os.path.join(tempdir, 'cached_likelihoods_subhmm_full')


def hash_library_and_data(mus,sigmas,data):
    thehash = hashlib.sha1(data)

    for mu, sigma in zip(mus,sigmas):
        thehash.update(mu)
        thehash.update(sigma)

    return thehash.hexdigest()

def split_data(big_data_array,model,num_parts):
    '''
    returns (and caches) datas and aBls, as per Issue #51.
    call it in place of np.array_split
    '''
    if not os.path.exists(full_likelihood_caches):
        os.mkdir(full_likelihood_caches)

    if hasattr(model, "obs_distnss"):
        filename = hash_library_and_data(
                [o.mu for o in model.obs_distnss[0]],
                [o.sigma for o in model.obs_distnss[0]],
                big_data_array)
    elif hasattr(model, "obs_distns"):
        filename = hash_library_and_data(
                [c.mu for c in model.obs_distns[0].components],
                [c.sigma for c in model.obs_distns[0].components],
                big_data_array)
    filepath = os.path.join(full_likelihood_caches,filename)

    if os.path.isfile(filepath):
        with open(filepath,'r') as infile:
            big_aBl_array = cPickle.load(infile)
        print 'loaded ALL aBls from cache: %s' % filepath
    else:
        model.add_data(data=big_data_array)
        if hasattr(model.states_list[0], 'aBls'):
            big_aBl_array = model.states_list.pop().aBls[0]
        else:
            big_aBl_array = model.states_list.pop().aBl
        with open(filepath,'w') as outfile:
            cPickle.dump(big_aBl_array,outfile,protocol=-1)
        print 'computed ALL aBls and saved to cache: %s' % filepath

    return np.array_split(big_data_array,num_parts), np.array_split(big_aBl_array,num_parts)

