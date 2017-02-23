# coding=utf-8
import seaborn.apionly as sns
import matplotlib.pyplot as plt
from lax import variables

def plot(df, cut_name,
         my_variables=False, save=False):

    if my_variables is None:
        my_variables = variables.get_variables()

    df_reduced = variables.reduce_df(df,
                                     my_variables)
    print('%s: %d of %d events not shown out of plotting window' % (cut_name,
                                                                     df['x'].count() - df_reduced['x'].count(),
                                                                     df['x'].count()))

    name = '%s Cut' % cut_name
    df_reduced[name] = ['Pass' if x == True else 'Fail' for x in df_reduced[cut_name]]

    number_passing = df_reduced[cut_name].sum()
    total = df_reduced['x'].count()


    if number_passing == total:
        print('Not plotting, no removed events for', cut_name)
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
    g.map_diag(plt.hist, histtype="step", linewidth=3)
    g.add_legend()

    for i, ax in enumerate(g.axes.flat):
        ax.set_xlim(my_variables[keys[i % len(my_variables)]]['range'])
        ax.set_ylim(my_variables[keys[int(i / len(my_variables))]]['range'])

    if save:
        for extension in ['pdf', 'png', 'eps']:
            plt.savefig('plots/%s_%d.%s' % (cut_name,
                                             len(my_variables),
                                            extension),
                        bbox_inches='tight')
    else:
        plt.show()

