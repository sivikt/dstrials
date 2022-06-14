import math

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.subplots import make_subplots
import plotly.graph_objects as go


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


def pies_grid(data, features_names, grid_cols: int = 3, pie_kws=None, subplots_kws=None):
    subplots_kws = subplots_kws if subplots_kws else {}
    pie_kws = pie_kws if pie_kws else dict(autopct='%.2f%%', fontsize=11, colormap='Pastel2_r')
    
    
    if 'figsize' not in subplots_kws:
        subplots_kws['figsize'] = (18,30)
        
    fig, axes = plt.subplots(math.ceil(len(features_names)/grid_cols), grid_cols, **subplots_kws)
    axes = [ax for axes_rows in axes for ax in axes_rows]

    for i in range(len(axes)-len(features_names)):
        fig.delaxes(axes[-i-1])

    for i, c in enumerate(features_names):        
        data[c].value_counts()[::-1].plot(
            kind='pie',
            ax=axes[i],
            title=c,
            **pie_kws
        )

        axes[i].set_ylabel('')
        
    fig.tight_layout()

    
def paired_scatter_plots(data, features_names, group_sz, scatter_plot_kws=None, subplots_kws=None):
    subplots_kws = subplots_kws if subplots_kws else {}
    scatter_plot_kws = scatter_plot_kws if scatter_plot_kws else {}
    
    if 'figsize' not in subplots_kws:
        subplots_kws['figsize'] = (10,10)

    if 'palette' not in scatter_plot_kws:
        scatter_plot_kws['palette'] = UNIFORM_COLORS
        
    feats_cnt = len(features_names)    
    plots_num = feats_cnt + int((feats_cnt**2 - feats_cnt)/2)
    fig, axes = plt.subplots(math.ceil(plots_num/group_sz), group_sz, **subplots_kws)
    axes = [ax for axes_rows in axes for ax in axes_rows]

    for i in range(len(axes)-plots_num):
        fig.delaxes(axes[-i-1])
    
    a = 0
    for i, f1 in enumerate(features_names):
        for f2 in features_names[i:]:
            sns.scatterplot(ax=axes[a], data=data, x=f1, y=f2, **scatter_plot_kws)
            a += 1
    
    #fig.tight_layout()
    
#             g = sns.PairGrid(
#                 data, 
#                 x_vars=group2, 
#                 y_vars=group1,  
#                 **pair_grid_kws
#             )

#             g.map_diag(sns.histplot)
#             g.map_lower(sns.scatterplot)
#             #g.map_lower(sns.regplot, scatter_kws={'alpha':0.3})

#             g.add_legend()

def plx_bars(data, x_features_names, y_name, cols=2):
    rows=5
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=[f"Median - {f}" for f in x_features_names]
    )

    for i, f in enumerate(x_features_names):
        r = int(i/cols)
        c = i - r*cols 
        fig.add_bar(
            x=data[f], 
            y=data[y_name],
            marker=dict(color=[4, 5, 6], coloraxis="coloraxis"),
            orientation='h', 
            row=r+1, col=c+1
        )

    fig.update_layout(
        title_text="Median values for numeric features across target",
        showlegend=False,
        height=900
    )
    fig.update_coloraxes(showscale=False)
    return fig


