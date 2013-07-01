from __future__ import division
import numpy as np
import timeit, time, textwrap

def run_speed_test(N=10000,D=200):
    ### logaddexp

    setup = \
    '''
    import numpy as np
    weights = np.random.random(size={D})
    log_likelihoods = np.log(np.random.random(size=({N},{D})))
    '''.format(D=D,N=N)

    stmt = \
    '''
    np.logaddexp.reduce(log_likelihoods + np.log(weights),axis=1)
    '''

    print 'logaddexp'
    my_timeit(setup,stmt)
    print ''

    ### manual logaddexp with dot

    stmt = \
    '''
    np.log(np.exp(log_likelihoods - log_likelihoods.max(1)[:,None]).dot(weights))
    '''

    print 'manual logaddexp with dot'
    my_timeit(setup,stmt)
    print ''

    ### dot with precomputed exp's

    setup = \
    '''
    import numpy as np
    weights = np.random.random(size={D})
    log_likelihoods = np.log(np.random.random(size=({N},{D})))
    maxes = log_likelihoods.max(1)
    shifted_likelihoods = np.exp(log_likelihoods - maxes[:,None])
    '''.format(D=D,N=N)

    stmt = \
    '''
    np.log(shifted_likelihoods.dot(weights)) + maxes
    '''

    print 'dot with precomputed exps'
    my_timeit(setup,stmt)
    print ''

def my_timeit(setup,stmt,repeat=3,precision=3):
    # emulates functionality as timeit.main()

    t = timeit.Timer(textwrap.dedent(stmt),textwrap.dedent(setup),time.time)

    for i in range(1,10):
        number = 10**i
        x = t.timeit(number)
        if x >= 0.2:
            break

    r = t.repeat(repeat,number)
    best = min(r)
    usec = best*1e6 / number

    print '%d loops,' % number,

    scales = np.array([1,1e3,1e6])
    idx = (scales < usec).sum() - 1
    units = ['usec','msec','sec']

    print 'best of %d: %.*g %s per loop' % (repeat, precision, usec / scales[idx], units[idx])

