from __future__ import division
from IPython.parallel import Client
from IPython.parallel.util import interactive

c = Client()
dv = c.direct_view()
lbv = c.load_balanced_view()

alldata = {}
alllikelihoods = {}

@lbv.parallel(block=True)
@interactive
def build_hsmm_states((data_id,left_censoring)):
    global global_model
    global alldata, alllikelihoods

    global_model.add_data(data=alldata[data_id],
            precomputed_likelihoods=alllikelihoods[data_id],
            initialize_from_prior=False,
            left_censoring=left_censoring)
    s = global_model.states_list.pop()
    stateseq, stateseq_norep, durations = s.stateseq, s.stateseq_norep, s.durations

    return (data_id, left_censoring, stateseq, stateseq_norep, durations)

