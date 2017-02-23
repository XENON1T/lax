"""Lichens grows on trees

Extend the Minitree produced DataFrames with derivative values.
"""
# -*- coding: utf-8 -*-

from lax.plotting import plot
from lax.variables import check_variable_list
import pandas as pd
import numpy as np
from collections import OrderedDict

pd.set_option('display.expand_frame_repr', False)


class Lichen(object):
    version = np.NaN

    def describe(self):
        print(self.__doc__)

    def pre(self, df):
        return df

    def process(self, df):
        df = self.pre(df)
        df = self._process(df)
        df = self.post(df)

        return df

    def _process(self, df):
        raise NotImplementedError()

    def post(self, df):
        if 'temp' in df.columns:
            return df.drop('temp', 1)
        return df


class RangeLichen(Lichen):
    allowed_range = None  # tuple of min then max
    variable = None  # variable name in DataFrame

    def get_allowed_range(self):
        if self.allowed_range is None:
            raise NotImplemented()

    def get_min(self):
        if self.variable is None:
            raise NotImplemented()
        return self.allowed_range[0]

    def get_max(self):
        if self.variable is None:
            raise NotImplemented()
        return self.allowed_range[0]

    def _process(self, df):
        df.loc[:, self.__class__.__name__] = (df[self.variable] > self.allowed_range[0]) & (
        df[self.variable] < self.allowed_range[1])
        return df


class ManyLichen(Lichen):
    lichen_list = []
    plots = False
    variables = None

    def get_cut_names(self):
        return [lichen.__class__.__name__ for lichen in self.lichen_list]

    def process(self, df):
        df.loc[:, (self.__class__.__name__)] = True

        for lichen in self.lichen_list:
            # Heavy lifting here
            df = lichen.process(df)

            cut_name = lichen.__class__.__name__

            if self.plots:
                plot(df[df[self.__class__.__name__]],
                     cut_name, self.variables)

            df.loc[:, self.__class__.__name__] = df[self.__class__.__name__] & df[cut_name]

        return df

    def debug(self,
              plots=True,
              variables=None):
        """Turn on debugging output (e.g. plots)

        :param plots: True or False on whether to make plots
        :param variables: List variables to plot in the format.  To specify ranges, see example in lax/variables.py.
        :return: None
        """
        if isinstance(plots, bool):
            self.plots = plots
        else:
            raise TypeError()

        if variables is None:  # Don't override if not specified
            pass
        else:
            self.variables = OrderedDict(check_variable_list(variables))
