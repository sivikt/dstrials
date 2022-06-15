import scipy.stats as ss
import numpy as np
import pandas as pd


def cramers_v(x, y):
    """Implements Cramér's V (sometimes referred to as Cramér's phi and denoted as φc). It is
    a measure of association between two nominal variables, giving a value between 0 and +1 (inclusive). 
    It is based on Pearson's chi-squared statistic and was published by Harald Cramér in 1946.
    
    It is symmetric V(x,y) = V(y,x).
    
    Notes
    -----
    Cramér's V can be a heavily biased estimator of its population counterpart and will tend to
    overestimate the strength of association. So a bias correction is needed.
    
    Implementation was found on StackOverflow
    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
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
    """ Uncertainty Coefficient. It is based on the conditional entropy between x and y.
    https://en.wikipedia.org/wiki/Uncertainty_coefficient
    
    It is asymmetric, meaning U(x,y) ≠ U(y,x).
    
    Notes
    -----
    Implementation was found on StackOverflow
    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    """
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
    

def correlation_ratio(categories, measurements):
    """The correlation ratio is a measure of the curvilinear relationship between the statistical
    dispersion within individual categories and the dispersion across the whole population or sample.
    The measure is defined as the ratio of two standard deviations representing these types of variation.
    https://en.wikipedia.org/wiki/Correlation_ratio
    
    Answers to the question "Given a continuous number, how well can you know to which category it belongs to?".
    
    Notes
    -----
    Implementation was found on StackOverflow
    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    """
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta


def correlation_ratio_pairwise(data, categories_cols, measurements_cols):       
    etas = np.zeros((len(categories_cols), len(measurements_cols)), dtype=np.float)
    
    for i, c1 in enumerate(categories_cols):
        for j, c2 in enumerate(measurements_cols):
            etas[i][j] = correlation_ratio(data[c1], data[c2])
            
    return etas
