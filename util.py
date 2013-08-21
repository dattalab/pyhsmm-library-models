from __future__ import division
import numpy as np
from collections import defaultdict
from itertools import count

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

