# -*- coding: utf-8 -*-
import numpy as np
from pax import units, configuration
PAX_CONFIG = configuration.load_configuration('XENON1T')
from lax.lichen import Lichen, RangeLichen, ManyLichen

class AllCutsSR0(ManyLichen):
    def __init__(self):
        self.lichen_list = [S1Threshold(),
                            S2Threshold(),
                            cS2Threshold(),
                            Fiducial(),
                            InteractionPeaksBiggest(),
                            S2AreaFractionTop(),
                            DoubleScatterS2(),
                            S2Width()]



class S2Threshold(RangeLichen):
    allowed_range = (100, np.inf)
    variable = 's2'


class cS2Threshold(S2Threshold):
    allowed_range = (100, np.inf)
    variable = 'cs2'


class S1Threshold(RangeLichen):
    allowed_range = (3, 70)
    variable = 'cs1'

class Fiducial(ManyLichen):
    """Fiducial volume cut.

    ! This is a 1T test FV - Updated 20/2/2017 !

    The fidicual volume cut defines the region in depth and radius that we
    trust and use for the exposure. This is the region where the background
    distribution is flat. 

    This version of the cut is based pax v6.2 bg run 0 data. See the
    note first results fiducial volume note for the study of the definition.

    Author: Sander breur sanderb@nikhef.nl

    """

    def __init__(self):
        self.lichen_list = [self.Z(),
                            self.R()]

    class Z(RangeLichen):
        allowed_range = (-83.45,-13.45)
        variable = 'z'

    class R(RangeLichen):
        variable = 'temp'

        def pre(self, df):
            df[self.variable] = np.sqrt(df['x']*df['x'] + df['y']*df['y'])
            return df

        allowed_range = (0, 39.85)


class S2AreaFractionTop(RangeLichen):
    allowed_range = (0.6, 0.72)
    variable = 's2_area_fraction_top'


class InteractionPeaksBiggest(ManyLichen):
    def __init__(self):
        self.lichen_list = [self.S1(),
                            self.S2()]

    class S1(Lichen):
        def _process(self, df):
            df[self.__class__.__name__] = df.s1 > df.largest_other_s1
            return df

    class S2(Lichen):
        def _process(self, df):
            df[self.__class__.__name__] = df.s2 > df.largest_other_s2
            return df


class DoubleScatterS2(Lichen):
    allowed_range = (0, np.inf)
    variable = 'temp'

    def other_s2_bound(self, s2):
        return np.clip((2 * s2) ** 0.5, 70, float('inf'))

    def _process(self, df):
        df[self.__class__.__name__] = df.largest_other_s2 < self.other_s2_bound(df.s2)
        return df

class S2Width(ManyLichen):
    """S2 Width cut modeling the difussion.

    The S2 width cut compares the S2 width to what we could expect based on its
    depth in the detector.  The inputs to this are the drift velocity and the
    diffusion constant.

    This version of the cut is based on quantiles on the Rn220 data.  See the
    note XXX for the study of the definition.  It should be applicable to data
    regardless of if it is ER or NR.

    Author: XXX yy@zz.nl

    """


    def __init__(self):
        self.lichen_list = [self.WidthHigh(),
                            self.WidthLow()]

    def s2_width_model(self, z):
        diffusion_constant = PAX_CONFIG['WaveformSimulator']['diffusion_constant_liquid']
        v_drift = PAX_CONFIG['DEFAULT']['drift_velocity_liquid']

        w0 = 304 * units.ns
        return np.sqrt(w0**2 - 3.6395 * diffusion_constant * z / v_drift**3)


    def subpre(self, df):
        # relative_s2_width
        df['temp'] = df['s2_range_50p_area'] / S2Width.s2_width_model(self, df['z'])
        return df

    def relative_s2_width_bounds(s2, kind='high'):
        x = 0.3 * np.log10(np.clip(s2, 150, 7000))
        if kind == 'high':
            return 2.3 - x
        elif kind == 'low':
            return -0.3 + x
        raise ValueError("kind must be high or low")


    class WidthHigh(Lichen):
        def pre(self, df):
            return S2Width.subpre(self, df)

        def _process(self, df):
            df[self.__class__.__name__] = (df.temp <= S2Width.relative_s2_width_bounds(df.s2,
                                                                                       kind='high'))
            return df

    class WidthLow(RangeLichen):
        def pre(self, df):
            return S2Width.subpre(self, df)

        def _process(self, df):
            df[self.__class__.__name__] = (S2Width.relative_s2_width_bounds(df.s2,
                                                                            kind='low') <= df.temp)
            return df
