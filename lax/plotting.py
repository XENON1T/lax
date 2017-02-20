# coding=utf-8
import seaborn as sns
import matplotlib.pyplot as plt
from lax import variables

save_file = True


def plot(df, cut_name, verbose=False, save=False):
    my_variables = variables.get_variables(verbose)

    df_reduced = variables.reduce_df(df, my_variables)
    print('%d of %d events not shown out of event window' % (df['x'].count() - df_reduced['x'].count(),
                                                             df['x'].count()))

    name = '%s Cut' % cut_name
    df_reduced[name] = ['Pass' if x == True else 'Fail' for x in df_reduced[cut_name]]

    number_passing = df_reduced[cut_name].sum()
    total = df_reduced[cut_name].count()

    if number_passing == total:
        return

    keys = list(my_variables.keys())

    g = sns.PairGrid(df_reduced,
                     vars=keys,
                     diag_sharey=False,
                     hue_order=['Pass', 'Fail'],
                     palette=['green', 'red', ],
                     hue=name,
                     hue_kws={"cmap": ["Greens", "Reds", ],
                              "marker": ["o", "x"]})
    g.map_lower(sns.kdeplot, shade=True,
                n_levels=10,
                shade_lowest=False, alpha=0.5, )
    g.map_upper(plt.scatter, alpha=0.2)
    # g.map_diag(sns.kdeplot, lw=3)
    g.map_diag(plt.hist, histtype="step", linewidth=3)
    g.add_legend()

    for i, ax in enumerate(g.axes.flat):
        ax.set_xlim(my_variables[keys[i % len(my_variables)]]['range'])
        ax.set_ylim(my_variables[keys[int(i / len(my_variables))]]['range'])

    if save_file:
        plt.savefig('plots/%s_%d.pdf' % (name, len(my_variables)), bbox_inches='tight')
        plt.savefig('plots/%s_%d.png' % (name, len(my_variables)), bbox_inches='tight')
    else:
        plt.show()

    df_reduced = df_reduced[df_reduced[cut_name] == True]
