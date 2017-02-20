"""Lichens grows on trees

Extend the Minitree produced DataFrames with derivative values.
"""
# -*- coding: utf-8 -*-

from lax import plotting

class Lichen(object):


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
        df[self.__class__.__name__] = (df[self.variable] > self.allowed_range[0]) & (
        df[self.variable] < self.allowed_range[1])
        return df


class ManyLichen(Lichen):
    lichen_list = []
    plots = False
    verbose = False

    def get_cut_names(self):
        return [lichen.__class__.__name__ for lichen in self.lichen_list]

    def process(self, df):
        all_cuts_bool = None
        for lichen in self.lichen_list:
            df = lichen.process(df)

            cut_name = lichen.__class__.__name__

            if all_cuts_bool is None:
                all_cuts_bool = df[cut_name]
            else:
                all_cuts_bool = all_cuts_bool & df[cut_name]



            if self.plots:
                self.plot(df[df[all_cuts_bool] == True], cut_name, self.verbose)

        df[self.__class__.__name__] = all_cuts_bool
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
