import seaborn as sns


def _ordered_countplot(y, **kwargs):
    ax = sns.countplot(y=y, palette="husl", order=y.value_counts().index)
    ax.bar_label(ax.containers[0])


def _barplot(x, y, **kwargs):
    ax = sns.barplot(x=x, y=y, orient='h', palette="husl", order=y.value_counts().index)
    ax.bar_label(ax.containers[0])


def _boxplot(x, **kwargs):
    sns.boxplot(x=x, palette="husl")
    

def _distplot(*args, **kwargs):    
    data = kwargs['data']
    var_val = data[args[0]].iloc[0]
    value_col = args[1]
    
    feature_plot_kws = kwargs.get('feature_plot_kws', None)
    feature_plot_kws = feature_plot_kws.get(var_val, None) if feature_plot_kws else None
    
    if feature_plot_kws:           
        sns.histplot(data[value_col], palette="Set1", color=kwargs['color'], **feature_plot_kws)
    else:
        sns.histplot(data[value_col], kde=True, palette="Set1")
        

def ordered_countplot_facet_grid(data, plots_for, values_col, **kwargs):
    g = sns.FacetGrid(data, col=plots_for, **kwargs)
    g.map(_ordered_countplot, values_col)
    #g.fig.suptitle(title)
    return g


def perc_barplot_facet_grid(data, plots_for, x_col, y_col, **kwargs):
    g = sns.FacetGrid(data, col=plots_for, xlim=(0, 100), **kwargs)
    g.map(_barplot, y_col, x_col)
    return g
    

def boxplot_facet_grid(data, plots_for, values_col, **kwargs):
    g = sns.FacetGrid(data, col=plots_for, **kwargs)
    g.map(_boxplot, values_col)
    return g


def distplot_facet_grid(data, plots_for, values_col, feature_plot_kws=None, facet_kws=None):
    if facet_kws:
        g = sns.FacetGrid(data, col=plots_for, **facet_kws)
    else:
        g = sns.FacetGrid(data, col=plots_for)
        
    if feature_plot_kws:
        g.map_dataframe(_distplot, plots_for, values_col, feature_plot_kws=feature_plot_kws)
    else:
        g.map_dataframe(_distplot, plots_for, values_col)
        
    return g


"""
import matplotlib.pyplot as plt
 
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
 
fig.suptitle('Geeksforgeeks - 2 x 3 axes Box plot with data')
 
iris = sns.load_dataset("iris")

sns.boxplot(ax=axes[0, 0], data=iris, x='species', y='petal_width')
sns.boxplot(ax=axes[0, 1], data=iris, x='species', y='petal_length')
sns.boxplot(ax=axes[0, 2], data=iris, x='species', y='sepal_width')
sns.boxplot(ax=axes[1, 0], data=iris, x='species', y='sepal_length')
sns.boxplot(ax=axes[1, 1], data=iris, x='species', y='petal_width')
sns.boxplot(ax=axes[1, 2], data=iris, x='species', y='petal_length')
"""