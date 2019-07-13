from __future__ import division

def bootstrap(x1, x2=None, paired=False, statfunction=None, smoothboot=False,
	alpha_level=0.05, reps=5000):
	
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.stats import norm
    from numpy.random import randint
    from scipy.stats import ttest_1samp, ttest_ind, ttest_rel
    from scipy.stats import mannwhitneyu, wilcoxon, norm
    import warnings


	

    # Turn to pandas series.
    x1 = pd.Series(x1).dropna()
    diff = False

    # Initialise statfunction
    if statfunction == None:
        statfunction = np.mean


    # Compute two-sided alphas.
    if alpha_level > 1. or alpha_level < 0.:
        raise ValueError("alpha_level must be between 0 and 1.")
    alphas = np.array([alpha_level/2., 1-alpha_level/2.])

    
    sns_bootstrap_kwargs = {'func': statfunction,
    
        'n_boot': reps,
    
        'smooth': smoothboot}

    if paired:
        # check x2 is not None:
        #if x2 is None:
            #raise ValueError('Please specify x2.')
        x2 = pd.Series(x2).dropna()
        if len(x1) != len(x2):
            raise ValueError('x1 and x2 are not the same length.')

    if (x2 is None) or (paired is True):
        if x2 is None:
            tx = x1
            paired = False
            ttest_single = ttest_1samp(x1, 0)[1]
            ttest_2_ind = 'NIL'
            ttest_2_paired = 'NIL'
            wilcoxonresult = 'NIL'

        elif paired is True:
            diff = True
            tx = x2 - x1
            ttest_single = 'NIL'
            ttest_2_ind = 'NIL'
            ttest_2_paired = ttest_rel(x1, x2)[1]
            wilcoxonresult = wilcoxon(x1, x2)[1]
        mannwhitneyresult = 'NIL'

        # Turns data into array, then tuple.
        tdata = (tx,)


        # The value of the statistic function applied
        # just to the actual data.
        summ_stat = statfunction(*tdata)
        statarray = sns.algorithms.bootstrap(tx, **sns_bootstrap_kwargs)
        statarray.sort()


        # Get Percentile indices
        pct_low_high = np.round((reps-1) * alphas)
        pct_low_high = np.nan_to_num(pct_low_high).astype('int')

    # Get Bias-Corrected Accelerated indices convenience function invoked.
    bca_low_high = bca(tdata, alphas, statarray,
        statfunction, summ_stat, reps)


    # Warnings for unstable or extreme indices.
    for ind in [pct_low_high, bca_low_high]:
        if np.any(ind == 0) or np.any(ind == reps-1):
            warnings.warn("Some values used extremal samples;"
            " results are probably unstable.")
        elif np.any(ind<10) or np.any(ind>=reps-10):
            warnings.warn("Some values used top 10 low/high samples;"
            " results may be unstable.")


    #summary = summ_stat
    is_paired = paired
    is_difference = diff
    statistic = str(statfunction)
    n_reps = reps
    ci = (1-alpha_level)*100
    stat_array = np.array(statarray)
    pct_ci_low = statarray[pct_low_high[0]]
    pct_ci_high = statarray[pct_low_high[1]]
    pct_low_high_indices = pct_low_high
    bca_ci_low = statarray[bca_low_high[0]]
    bca_ci_high = statarray[bca_low_high[1]]
    bca_low_high_indices = bca_low_high
    pvalue_1samp_ttest = ttest_single
    pvalue_2samp_ind_ttest = ttest_2_ind
    pvalue_2samp_paired_ttest = ttest_2_paired
    pvalue_wilcoxon = wilcoxonresult
    pvalue_mann_whitney = mannwhitneyresult

    return ci, bca_ci_low, bca_ci_high, pvalue_2samp_paired_ttest



def jackknife_indexes(data):
    # Taken without modification from scikits.bootstrap package.
    """
    From the scikits.bootstrap package.
    Given an array, returns a list of arrays where each array is a set of
    jackknife indexes.

    For a given set of data Y, the jackknife sample J[i] is defined as the
    data set Y with the ith data point deleted.
    """
    import numpy as np

    base = np.arange(0,len(data))
    return (np.delete(base,i) for i in base)

def bca(data, alphas, statarray, statfunction, ostat, reps):
    '''
    Subroutine called to calculate the BCa statistics.
    Borrowed heavily from scikits.bootstrap code.
    '''
    import warnings

    import numpy as np
    import pandas as pd
    import seaborn as sns

    from scipy.stats import norm
    from numpy.random import randint

    # The bias correction value.
    z0 = norm.ppf( ( 1.0*np.sum(statarray < ostat, axis = 0)  ) / reps )

    # Statistics of the jackknife distribution
    jackindexes = jackknife_indexes(data[0])
    jstat = [statfunction(*(x[indexes] for x in data))
            for indexes in jackindexes]
    jmean = np.mean(jstat,axis = 0)

    # Acceleration value
    a = np.divide(np.sum( (jmean - jstat)**3, axis = 0 ),
        ( 6.0 * np.sum( (jmean - jstat)**2, axis = 0)**1.5 )
        )
    if np.any(np.isnan(a)):
        nanind = np.nonzero(np.isnan(a))
        warnings.warn("Some acceleration values were undefined."
        "This is almost certainly because all values"
        "for the statistic were equal. Affected"
        "confidence intervals will have zero width and"
        "may be inaccurate (indexes: {})".format(nanind))
    zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)
    avals = norm.cdf(z0 + zs/(1-a*zs))
    nvals = np.round((reps-1)*avals)
    nvals = np.nan_to_num(nvals).astype('int')

    return nvals




