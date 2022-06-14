import scipy.stats as ss
import numpy as np
import pandas as pd


def cramers_v(x, y):
    """Implements Cramér's V (sometimes referred to as Cramér's phi and denoted as φc). It is a measure of association 
    between two nominal variables, giving a value between 0 and +1 (inclusive). 
    It is based on Pearson's chi-squared statistic and was published by Harald Cramér in 1946.
    
    Notes
    -----
    Cramér's V can be a heavily biased estimator of its population counterpart and will tend to overestimate the strength
    of association. So a bias correction is needed.
    
    Implementation was found on StackOverflow
    """
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def cramers_v_pairwise(data, columns1, columns2=None):
    if columns2 is None:
        columns2 = columns1
        
    cramers_v_rels = np.zeros((len(columns1), len(columns2)), dtype=np.float)
    
    for i, c1 in enumerate(columns1):
        for j, c2 in enumerate(columns2):
            cramers_v_rels[i][j] = cramers_v(data[c1], data[c2])
            
    return cramers_v_rels
        
        
def theils_u(x, y):
    """Implementation was found on StackOverflow"""
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x