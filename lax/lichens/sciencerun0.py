# -*- coding: utf-8 -*-
import numpy as np
from pax import units, configuration

PAX_CONFIG = configuration.load_configuration('XENON1T')
from lax.lichen import Lichen, RangeLichen, ManyLichen
from lax import __version__ as lax_version


class AllCuts(ManyLichen):
    version = lax_version

    def __init__(self):
        self.lichen_list = [
            FiducialCylinder1T(),
            InteractionExists(),
            S2Threshold(),
            InteractionPeaksBiggest(),
            S2AreaFractionTop(),
            S2SingleScatter(),
            SignalOverPreS2Junk(),
        ]


class LowEnergyCuts(AllCuts):
    def __init__(self):
        AllCuts.__init__(self)
        self.lichen_list[1] = S1LowEnergyRange()
        self.lichen_list.append(S2Width())


class InteractionExists(RangeLichen):
    """Check that an interaction found

    Not studied at all.
    """
    version = 0
    allowed_range = (0, np.inf)
    variable = 'cs1'


class S2Threshold(RangeLichen):
    """The S2 energy at which the trigger is perfectly efficient.

    See: xenon:xenon1t:aalbers:preliminary_trigger_settings
    """
    version = 0
    allowed_range = (150, np.inf)
    variable = 's2'


class S1LowEnergyRange(RangeLichen):
    version = 0
    allowed_range = (0, 200)
    variable = 'cs1'


class FiducialCylinder1T(ManyLichen):
    """Fiducial volume cut.

    ! This is a 1T test FV - Updated 20/2/2017 !

    The fidicual volume cut defines the region in depth and radius that we
    trust and use for the exposure. This is the region where the background
    distribution is flat.

    This version of the cut is based pax v6.2 bg run 0 data. See the
    note first results fiducial volume note for the study of the definition.

    Author: Sander breur sanderb@nikhef.nl

    """
    version = 2

    def __init__(self):
        self.lichen_list = [self.Z(),
                            self.R()]

    class Z(RangeLichen):
        allowed_range = (-83.45, -13.45)
        variable = 'z'

    class R(RangeLichen):
        variable = 'r'

        def pre(self, df):
            df[self.variable] = np.sqrt(df['x'] * df['x'] + df['y'] * df['y'])
            return df

        allowed_range = (0, 39.85)


class S2AreaFractionTop(RangeLichen):
    """Blah

    Don't use above XXXX PE due to saturation.
    """
    version = 0

    allowed_range = (0.6, 0.72)
    variable = 's2_area_fraction_top'


class InteractionPeaksBiggest(ManyLichen):
    version = 0

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


class SignalOverPreS2Junk(Lichen):
    """Compare S1 and S2 area to the area of other peaks before interaction S2

    This cut value is made up.... or at least found in a random notebook.
    """
    version = 0
    allowed_range = (0.5, 10)
    variable = 'signal_over_pre_s2_junk'

    def pre(self, df):
        df['signal_over_pre_s2_junk'] = (df.s2 + df.s1) / (df.area_before_main_s2)
        return df


class S2SingleScatter(Lichen):
    version = 0
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

    Only use below XXX PE to avoid track length effects.

    Author: XXX yy@zz.nl

    """
    version = 0

    def __init__(self):
        self.lichen_list = [self.S2WidthHigh(),
                            self.S2WidthLow()]

    def s2_width_model(self, z):
        diffusion_constant = PAX_CONFIG['WaveformSimulator']['diffusion_constant_liquid']
        v_drift = PAX_CONFIG['DEFAULT']['drift_velocity_liquid']

        w0 = 304 * units.ns
        return np.sqrt(w0 ** 2 - 3.6395 * diffusion_constant * z / v_drift ** 3)

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

    class S2WidthHigh(Lichen):
        def pre(self, df):
            return S2Width.subpre(self, df)

        def _process(self, df):
            df[self.__class__.__name__] = (df.temp <= S2Width.relative_s2_width_bounds(df.s2,
                                                                                       kind='high'))
            return df

    class S2WidthLow(RangeLichen):
        def pre(self, df):
            return S2Width.subpre(self, df)

        def _process(self, df):
            df[self.__class__.__name__] = (S2Width.relative_s2_width_bounds(df.s2,
                                                                            kind='low') <= df.temp)
            return df
