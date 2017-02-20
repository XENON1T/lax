# coding=utf-8

from collections import OrderedDict

VARIABLES = [
    ('r', {'range' : (0,50)}),
    ('z', {'range' : (-100, 0)}),
    ('s1', {'range' : (0,100)}),
    ('s2', {'range' : (0,1e4)}),
    ('cs1', {'range' : (0,100)}),
    ('cs2', {'range' : (0,1000)}),
    ('largest_other_s1', {'range' : (0, 100)}),
    ('largest_other_s2' , {'range' : (0,500)}),
    ('s1_range_50p_area' , {'range' : (0,10000)}),
    ('s2_range_50p_area' , {'range' : (0,100000)}),
    ( 's1_area_fraction_top', {'range' : (0,1)}),
    ('s2_area_fraction_top', {'range' : (0.4,0.9)}),
    ( 'area_before_main_s2' , {'range' : (0,2000)})
]

def get_variables(verbose=False):
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

