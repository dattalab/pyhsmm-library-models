from __future__ import division
from IPython.parallel import Client
from IPython.parallel.util import interactive

c = Client()
dv = c.direct_view()
lbv = c.load_balanced_view()

