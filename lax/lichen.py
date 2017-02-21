"""Lichens grows on trees

Extend the Minitree produced DataFrames with derivative values.
"""
# -*- coding: utf-8 -*-

from lax.plotting import plot
import numpy as np

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
        df.loc[:, self.__class__.__name__] = (df[self.variable] > self.allowed_range[0]) & (df[self.variable] < self.allowed_range[1])
        return df


class ManyLichen(Lichen):
    lichen_list = []
    plots = False
    verbose = False

    def get_cut_names(self):
        return [lichen.__class__.__name__ for lichen in self.lichen_list]

    def process(self, df):
        df[self.__class__.__name__] = True

        for lichen in self.lichen_list:
            # Heavy lifting here
            df = lichen.process(df)

            cut_name = lichen.__class__.__name__

            if self.plots:

                plot(df[df[self.__class__.__name__]],
                     cut_name,
                     self.verbose)

                df.loc[:, self.__class__.__name__] = df[self.__class__.__name__] & df[cut_name]

        return df

    def debug(self,
              plots=True,
              verbose=None):
        if isinstance(plots, bool):
            self.plots = plots
        else:
            raise TypeError()

        if isinstance(verbose, bool):
            self.verbose = verbose
        elif verbose is None:  # Don't override if not specified
            pass
        else:
            raise TypeError()
