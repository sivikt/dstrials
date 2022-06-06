import seaborn as sns


UNIFORM_COLORS = 'pastel'


def _ordered_countplot(y, **kwargs):
    ax = sns.countplot(y=y, palette=UNIFORM_COLORS, order=y.value_counts().index)
    ax.bar_label(ax.containers[0])


def _barplot(x, y, **kwargs):
    ax = sns.barplot(x=x, y=y, orient='h', palette=UNIFORM_COLORS, order=y.value_counts().index)
    ax.bar_label(ax.containers[0])


def _boxplot(x, **kwargs):
    sns.boxplot(x=x, palette=UNIFORM_COLORS)
    

def _distplot(*args, **kwargs):    
    data = kwargs['data']
    var_val = data[args[0]].iloc[0]
    value_col = args[1]
    
    feature_plot_kws = kwargs.get('feature_plot_kws', None)
    feature_plot_kws = feature_plot_kws.get(var_val, None) if feature_plot_kws else None
    
    if feature_plot_kws:           
        sns.histplot(data[value_col], palette=UNIFORM_COLORS, **feature_plot_kws)
    else:
        sns.histplot(data[value_col], kde=True, palette=UNIFORM_COLORS)
        

def ordered_countplot_facet_grid(data, plots_for, values_col, title=None, facet_kws=None):
    facet_kws = facet_kws if facet_kws else {}
        
    g = sns.FacetGrid(data, col=plots_for, **facet_kws)
    g.map(_ordered_countplot, values_col)
    
    g.set_axis_labels('count', 'category')
    
    if title:
        g.fig.subplots_adjust(top=0.93)
        g.fig.suptitle(title)
        
    return g


def perc_barplot_facet_grid(data, plots_for, x_col, y_col, title=None, facet_kws=None):
    facet_kws = facet_kws if facet_kws else {}
    
    g = sns.FacetGrid(data, col=plots_for, xlim=(0, 100), **facet_kws)
    g.map(_barplot, y_col, x_col)
    
    g.set_axis_labels('count', 'category')
    
    if title:
        g.fig.subplots_adjust(top=0.93)
        g.fig.suptitle(title)
        
    return g
    

def boxplot_facet_grid(data, plots_for, values_col, title=None, facet_kws=None):
    facet_kws = facet_kws if facet_kws else {}
    
    g = sns.FacetGrid(data, col=plots_for, **facet_kws)
    g.map(_boxplot, values_col)
    
    if title:
        g.fig.subplots_adjust(top=0.95)
        g.fig.suptitle(title)
        
    return g


def distplot_count_facet_grid(data, plots_for, values_col, title=None, feature_plot_kws=None, facet_kws=None):
    facet_kws = facet_kws if facet_kws else {}
    feature_plot_kws = feature_plot_kws if feature_plot_kws else {}
    
    g = sns.FacetGrid(data, col=plots_for, **facet_kws)
    g.map_dataframe(_distplot, plots_for, values_col, feature_plot_kws=feature_plot_kws)
    g.set_axis_labels('value', 'count')
        
    if title:
        g.fig.subplots_adjust(top=0.95)
        g.fig.suptitle(title)
        
    return g


def distplot_percent_facet_grid(data, plots_for, values_col, title=None, feature_plot_kws=None, facet_kws=None):
    facet_kws = facet_kws if facet_kws else {}
    feature_plot_kws = feature_plot_kws if feature_plot_kws else {}
    
    g = sns.FacetGrid(data, col=plots_for, **facet_kws)
    g.map_dataframe(_distplot, plots_for, values_col, feature_plot_kws=feature_plot_kws)
    g.set_axis_labels('value', 'cumulative percent')
    
    if title:
        g.fig.subplots_adjust(top=0.95)
        g.fig.suptitle(title)
        
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