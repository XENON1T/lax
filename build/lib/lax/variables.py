# coding=utf-8

from collections import OrderedDict

VARIABLES = [
    ('r', {'range': (0, 50)}),
    ('z', {'range': (-100, 0)}),
    ('s1', {'range': (0, 100)}),
    ('s2', {'range': (0, 1e4)}),
    ('cs1', {'range': (0, 100)}),
    ('cs2', {'range': (0, 1e4)}),
    ('largest_other_s1', {'range': (0, 100)}),
    ('largest_other_s2', {'range': (0, 500)}),
    ('s1_range_50p_area', {'range': (0, 100)}),
    ('s2_range_50p_area', {'range': (0, 3000)}),
    ('s1_area_fraction_top', {'range': (0, 1)}),
    ('s2_area_fraction_top', {'range': (0.4, 0.9)}),
    ('area_before_main_s2', {'range': (0, 2000)})
]


def check_variable_list(variables):
    """ Check formatting of variables string

    The formatting of variables can either be:

        ('r', 'z')

    Or:

        [('s1_area_fraction_top', {'range': (0, 1)}),
        ('s2', {'range': (0, 1e4)})]

    :param variables: List of variables to plot.
    :return:
    """
    if not isinstance(variables, (tuple, list)):
        raise ValueError('Variables must be tuple or list.')

    if len(variables) == 0:
        raise ValueError('Variables list cannot be empty.')

    new_variables = []
    for variable in variables:
        if isinstance(variable, str):
            temp = [x for x in VARIABLES if x[0] == variable]
            if len(temp) == 0:
                raise ValueError('Range must be specified for %s' % variable)
            new_variables.append(temp)
        elif isinstance(variable, (tuple, list)):
            if len(variables) != 2:
                raise ValueError('Misspecified range, see lax/variables.py')
            new_variables.append(variable)
    return variables


def get_variables(verbose=False):
    """

    :param verbose:
    :return:
    """
    if verbose:
        return OrderedDict(VARIABLES)
    return OrderedDict(VARIABLES[0:4])


def reduce_df(df, variables, squash=False):
    if squash:
        df = df.loc[::, variables.keys()]

    for key, value in variables.items():
        if 'range' not in value:
            continue
        df = df[(df[key] >= value['range'][0]) & (df[key] <= value['range'][1])]

    return df
