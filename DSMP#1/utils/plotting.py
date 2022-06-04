import seaborn as sns


def ordered_countplot(y, **kwargs):
    ax = sns.countplot(y=y, palette="husl", order=y.value_counts().index, **kwargs)
    ax.bar_label(ax.containers[0])


def barplot(x, y, **kwargs):
    ax = sns.barplot(x=x, y=y, orient='h', palette="husl", order=y.value_counts().index, **kwargs)
    ax.bar_label(ax.containers[0])


def ordered_countplot_facet_grid(data, plots_for, values_col, **kwargs):
    cat_fg = sns.FacetGrid(data, col=plots_for, **kwargs)
    cat_fg.map(barplot, values_col)
    

def barplot_facet_grid(data, plots_for, x_col, y_col, **kwargs):
    cat_fg = sns.FacetGrid(data, col=plots_for, xlim=(0, 100), **kwargs)
    cat_fg.map(barplot, y_col, x_col)
    