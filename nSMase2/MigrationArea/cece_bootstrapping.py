# Not sure what this package is, copied this import from DABEST
from __future__ import division

# This code is largely based onbootstrap tools in DABEST.

def bootstrap(x1, x2, paired=True, statfunction=None, smoothboot=False,
	alpha_level=0.05, reps=5000):
    '''
    Computes summary statistics and booststrapped confidence interval for
    paired data.

    Keywords:
    x1, x2: Paired 1D arrays

    paired: boolean, default True
        Whether x1 and x2 are paired samples

    statfunction: function
        Summary statistic to call on data. Default is np.mean

    alpha_level: float, default 0.05
        alpha = 0.05 gives 95 percent confidence interval

    reps: int, default = 5000
        number of bootstrap replicates

    Returns:

    dictionary of statistics






    '''
	
    # Imports
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
    # Calculates more statistics than it returns.
    # Function can be modified to return necessary statistics.
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

    stat_dict = {'ci' : ci, 'pct_ci_low' : pct_ci_low, 'pct_ci_high' : pct_ci_high, 'pct_low_high_indices' : pct_low_high_indices, 
    'bca_ci_low' : bca_ci_low, 'bca_ci_high' : bca_ci_high, 'bca_low_high_indices' : bca_low_high, 'pvalue_1samp_ttest' : pvalue_1samp_ttest, 
    'pvalue_2samp_ind_ttest' : pvalue_2samp_ind_ttest, 'pvalue_2samp_paired_ttest' : pvalue_2samp_paired_ttest, 
    'pvalue_wilcoxon' : pvalue_wilcoxon, 'pvalue_mann_whitney' : pvalue_mann_whitney}

    return stat_dict



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

def add_more_stats(stats_df, data_df):
    """
    Stat definitions:

        ci: float
            The size of the confidence interval reported (in percentage).

        pct_ci_low, pct_ci_high: floats
            The upper and lower bounds of the confidence interval as computed
            by taking the percentage bounds.

        pct_low_high_indices: array
            An array with the indices in `stat_array` corresponding to the
            percentage confidence interval bounds.

        bca_ci_low, bca_ci_high: floats
            The upper and lower bounds of the bias-corrected and accelerated
            (BCa) confidence interval. See Efron 1977.

        bca_low_high_indices: array
            An array with the indices in `stat_array` corresponding to the BCa
            confidence interval bounds.

        pvalue_1samp_ttest: float
            P-value obtained from scipy.stats.ttest_1samp. If 2 arrays were
            passed (x1 and x2), returns 'NIL'.
            See https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.stats.ttest_1samp.html

        pvalue_2samp_ind_ttest: float
            P-value obtained from scipy.stats.ttest_ind.
            If a single array was given (x1 only), or if `paired` is True,
            returns 'NIL'.
            See https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.stats.ttest_ind.html

        pvalue_2samp_related_ttest: float
            P-value obtained from scipy.stats.ttest_rel.
            If a single array was given (x1 only), or if `paired` is False,
            returns 'NIL'.
            See https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.stats.ttest_rel.html

        pvalue_wilcoxon: float
            P-value obtained from scipy.stats.wilcoxon.
            If a single array was given (x1 only), or if `paired` is False,
            returns 'NIL'.
            The Wilcoxons signed-rank test is a nonparametric paired test of
            the null hypothesis that the related samples x1 and x2 are from
            the same distribution.
            See https://docs.scipy.org/doc/scipy-1.0.0/reference/scipy.stats.wilcoxon.html

        pvalue_mann_whitney: float
            Two-sided p-value obtained from scipy.stats.mannwhitneyu.
            If a single array was given (x1 only), returns 'NIL'.
            The Mann-Whitney U-test is a nonparametric unpaired test of the null
            hypothesis that x1 and x2 are from the same distribution.
            See https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.stats.mannwhitneyu.html
    """
    # Check dataframe has "Treatment" column
    assert('Treatment' in stats_df.columns), 'Given dataframe doesn\'t have a \'Treatment\' column.'
    
    # Get list of unique treatments
    treatment_list = stats_df['Treatment']
    
    # List of possible stats the function can return
    stats_options = ['ci', 'pct_ci_low', 'pct_ci_high', 'pct_low_high_indices',
                'bca_ci_low', 'bca_ci_high', 'bca_low_high_indices', 
                 'pvalue_1samp_ttest', 'pvalue_2samp_ind_ttest', 
                'pvalue_2samp_paired_ttest', 'pvalue_wilcoxon',
                'pvalue_mann_whitney']
    
    # Get user input and check it looks ok
    
    # Show user list of possible stats
    print('Here are the possible statistics to add:')
    for stat in stats_options:
        print(stat)
    print('Check docstring for stat meanings.')
    print('\n')
    
    # Get list of stats the user wants to add to dataframe
    stats_to_add = input('What stats do you want? Separate with commas: ')
    stat_list = stats_to_add.split(', ')
    
    # Check statistics are in list of possible statistics
    for requested in stat_list:
        assert(requested in stats_options), 'Error! Given statistic not an option. Please check list of possible stats and try again.'
        print(requested)
    
    # Check with user that we are getting the correct statistics
    print('Are these the correct stats?')
    correct = input('Type \'y\' if correct or \'n\' if incorrect: ')
    assert(correct == 'y'), 'Incorrect stats selected. Please rerun the function and try again.'
    print('Adding stats to statistics dataframe')
    
    new_stats_dict = {}
    
    for statistic in stat_list:
        new_stats_dict[statistic] = []
    
    for treatment in treatment_list:
        # Extract data for each treatment
        df_treat = data_df.loc[data_df['Treatment'] == treatment]
        df_treat = df_treat.reset_index()
        
        x1 = df_treat['Norm Control Area']
        x2 = df_treat['Norm Experiment Area']
        
        stats_dict = bootstrap(x1, x2=x2, paired=True)
        
        for stat in stats_dict:
            if stat in stat_list:
                s = stats_dict[stat]
                new_stats_dict[stat].append(s)
    
    for key in new_stats_dict:
        stats_df[key] = new_stats_dict[key]
    
    return stats_df  



