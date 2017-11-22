"""The cuts for science run 1

This includes all current definitions of the cuts for the second science run
"""

# -*- coding: utf-8 -*-
import inspect
import os
import numpy as np
from pax import units

from lax.lichen import Lichen, RangeLichen, ManyLichen, StringLichen
from lax.lichens import sciencerun0
from lax import __version__ as lax_version

# Store the directory of our data files
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
                        '..', 'data')


class AllEnergy(ManyLichen):
    """Cuts applicable for low and high energy (gammas)

    This is a subset mostly of the low energy cuts.
    """
    version = lax_version

    def __init__(self):
        self.lichen_list = [
            FiducialCylinder1p3T(),
            InteractionExists(),
            S2Threshold(),
            S2AreaFractionTop(),
            S2SingleScatter(),
            DAQVeto(),
            S1SingleScatter(),
            S2PatternLikelihood(),
            S2Tails(),
            InteractionPeaksBiggest()
        ]


class LowEnergyRn220(AllEnergy):
    """Select Rn220 events with cs1<200

    This is the list that we use for the Rn220 data to calibrate ER in the
    region of interest.

    It doesn't contain the PreS2Junk cut
    """

    def __init__(self):
        AllEnergy.__init__(self)
        # Replaces Interaction exists
        self.lichen_list[1] = S1LowEnergyRange()

        # Use a simpler single scatter cut
        self.lichen_list[4] = S2SingleScatterSimple()

        self.lichen_list += [
            S1PatternLikelihood(),
            S2Width(),
            S1MaxPMT(),
            S1AreaFractionTop(),
        ]


class LowEnergyBackground(LowEnergyRn220):
    """Select background events with cs1<200

    This is the list that we'll use for the actual DM search. In addition to the
    LowEnergyRn220 list, it contains S2Tails and PreS2Junk.
    """

    def __init__(self):
        LowEnergyRn220.__init__(self)

        self.lichen_list += [
            PreS2Junk(),
        ]

class LowEnergyAmBe(LowEnergyRn220):
    """Select AmBe events with cs1<200 with appropriate cuts

    It is the same as the LowEnergyRn220 cuts.
    """

    def __init__(self):
        LowEnergyRn220.__init__(self)


class LowEnergyNG(LowEnergyRn220):
    """Select NG events with cs1<200 with appropriate cuts

    It is the same as the LowEnergyRn220 cuts.
    """

    def __init__(self):
        LowEnergyRn220.__init__(self)


DAQVeto = sciencerun0.DAQVeto


class S2Tails(Lichen):
    """Check if event is in a tail of a previous S2
    Requires S2Tail minitrees.
    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:subgroup:wimphysics:s2_tails_sr0 (SR0)
    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:subgroup:20170720_sr1_cut_s2_tail (SR1)
    Contact: Daniel Coderre <daniel.coderre@physik.uni-freiburg.de>
             Diego Ramírez <diego.ramirez@physik.uni-freiburg.de>
    """
    version = 1

    def _process(self, df):
        df.loc[:, self.name()] = (df['s2_over_tdiff'] < 0.025)
        return df

FiducialCylinder1T_TPF2dFDC = sciencerun0.FiducialCylinder1T_TPF2dFDC

FiducialCylinder1T = sciencerun0.FiducialCylinder1T

FiducialCylinder1p3T = sciencerun0.FiducialCylinder1p3T

FvConfigs = [
    # Mass (kg), (z0, vz, p, vr2)
    (1000, (-57.58, 31.25, 4.20, 1932.53)),
    (1025, (-57.29, 31.65, 3.71, 1987.85)),
    (1050, (-62.25, 33.89, 3.08, 1969.68)),
    (1075, (-59.73, 35.58, 2.97, 1938.27)),
    (1100, (-58.36, 36.97, 2.67, 1951.06)),
    (1125, (-58.57, 37.36, 2.78, 1953.64)),
    (1150, (-58.71, 37.37, 3.25, 1934.94)),
    (1175, (-57.35, 38.17, 3.14, 1944.21)),
    (1200, (-56.44, 39.23, 2.88, 1961.09)),
    (1225, (-55.81, 40.30, 2.80, 1969.58)),
    (1250, (-55.44, 40.70, 3.21, 1936.89)),
    (1275, (-54.62, 41.19, 3.29, 1937.40)),
    (1300, (-54.04, 42.08, 2.56, 2041.03)),
    (1325, (-53.31, 42.52, 2.85, 2005.24)),
    (1350, (-51.63, 43.64, 3.15, 1949.59)),
    (1375, (-51.85, 44.07, 2.87, 2003.40)),
    (1400, (-52.22, 43.88, 3.88, 1949.48)),
    (1425, (-50.87, 45.22, 2.75, 2046.11)),
    (1450, (-51.66, 43.60, 3.58, 2053.80)),
    (1475, (-51.99, 43.92, 3.88, 2044.43)),
    (1500, (-52.50, 43.69, 4.31, 2059.53)),
    (1525, (-51.51, 44.67, 3.68, 2093.78)),
    (1550, (-51.13, 45.04, 3.98, 2088.21)),
    (1575, (-49.44, 46.67, 3.43, 2091.66)),
    (1600, (-50.15, 45.95, 3.77, 2126.11)),
    (1625, (-50.06, 45.99, 4.32, 2119.37)),
    (1650, (-50.16, 45.95, 4.67, 2137.15)),
    (1675, (-49.10, 47.05, 4.53, 2128.00)),
    (1700, (-49.54, 46.51, 6.10, 2129.72)),
]

class FiducialTestEllips(StringLichen):
    """TESTFiducial volume cut using NN 3D FDC.
    Temporary/under development, for preliminary
    comparisons between the different masses. For every mass
    in the fv_config keys a FiducialTestEllips<mass> is made.
    For more info on the construction of the FV see:
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:analysis:sciencerun1:fiducial_volume:optimized_ellips
    sanderb@nikhef.nl
    """
    
    version = 1
    parameter_symbols = tuple('z0 vz p vr2'.split())
    parameter_values = None   # Will be tuple of parameter values
    string = "((( (((z_3d_nn-@z0)**2)**0.5) /@vz)**@p)+ (r_3d_nn**2/@vr2)**@p) < 1"
    
    def _process(self, df):
        bla = dict(zip(self.parameter_symbols, self.parameter_values))
        df.loc[:, self.name()] = df.eval(self.string,
                                         global_dict=bla)
        return df
    
    def pre(self, df):
        df.loc[:, 'r_3d_nn'] = np.sqrt(df['x_3d_nn']**2 + df['y_3d_nn']**2)
        return df

for mass, params in FvConfigs:
    name = 'FiducialTestEllips' + str(int(mass))
    c = type(name, (FiducialTestEllips,), dict())
    c.parameter_values = params
    locals()[name] = c

AmBeFiducial = sciencerun0.AmBeFiducial

class NGFiducial(StringLichen):
    """NG Fiducial volume cut.
    Early Implimentation og NG Fiducial Volume. Might not be Needed

    Link to Note:xenon:xenon1t:analysis:dominick:sr1_ambe_check (By Dominic)
    """
    version = 0
    string = "(distance_to_source < 111.5) & (-92.9 < z) & (z < -9) & (sqrt(x*x + y*y) < 42.00)"

    def pre(self, df):
        source_position = (31.6, 86.8, -50)
        df.loc[:, 'distance_to_source'] = ((source_position[0] - df['x']) ** 2 +
                                           (source_position[1] - df['y']) ** 2 +
                                           (source_position[2] - df['z']) ** 2) ** 0.5
        return df


InteractionExists = sciencerun0.InteractionExists

InteractionPeaksBiggest = sciencerun0.InteractionPeaksBiggest  # See PR #70

S1LowEnergyRange = sciencerun0.S1LowEnergyRange

S1MaxPMT = sciencerun0.S1MaxPMT

S1PatternLikelihood = sciencerun0.S1PatternLikelihood

S1SingleScatter = sciencerun0.S1SingleScatter

S2AreaFractionTop = sciencerun0.S2AreaFractionTop

S2SingleScatter = sciencerun0.S2SingleScatter

S2SingleScatterSimple = sciencerun0.S2SingleScatterSimple


class S2PatternLikelihood(StringLichen):
    """Reject poorly reconstructed S2s and multiple scatters.
    Details of the likelihood can be seen in the following note. Here, 95
    quantile acceptance line estimated with Rn220 data (pax_v6.8.0) is used.
       xenon:xenon1t:sim:notes:tzhu:pattern_cut_tuning
    Requires Extended minitrees.
    Contact: Bart Pelssers  <bart.pelssers@fysik.su.se> Tianyu Zhu <tz2263@columbia.edu>
    """
    version = 3
    string = "s2_pattern_fit < 0.0404*s2 + 594*s2**0.0737 - 686"


S2Threshold = sciencerun0.S2Threshold


class S2Width(ManyLichen):
    """S2 Width cut based on diffusion model
    The S2 width cut compares the S2 width to what we could expect based on its depth in the detector. The inputs to
    this are the drift velocity and the diffusion constant. The allowed variation in S2 width is greater at low
    energy (since it is fluctuating statistically) Ref: (arXiv:1102.2865)
    It should be applicable to data regardless of if it ER or NR;
    above cS2 = 1e5 pe ERs the acceptance will go down due to track length effects.
    Tune the diffusion model parameters based on pax v6.4.2 AmBe data according to note:
    xenon:xenon1t:yuehuan:analysis:0sciencerun_s2width_update0#comparison_with_diffusion_model_cut_by_jelle_pax_v642
    Contact: Yuehuan <weiyh@physik.uzh.ch>, Jelle <jaalbers@nikhef.nl>
    """
    version = 3

    def __init__(self):
        self.lichen_list = [self.S2WidthHigh(),
                            self.S2WidthLow()]

    @staticmethod
    def s2_width_model(z):
        diffusion_constant = 31.73 * ((units.cm)**2) / units.s
        v_drift = 1.335 * (units.um) / units.ns
        GausSigmaToR50 = 1.349

        EffectivePar = 0.925
        Sigma_0 = 229.58 * units.ns
        return GausSigmaToR50 * np.sqrt(Sigma_0 ** 2 - EffectivePar * 2 * diffusion_constant * z / v_drift ** 3)

    @staticmethod
    def subpre(self, df):
        df.loc[:, 'temp'] = df['s2_range_50p_area'] / S2Width.s2_width_model(df['z'])
        return df

    @staticmethod
    def relative_s2_width_bounds(s2, kind='high'):
        x = 0.5 * np.log10(np.clip(s2, 150, 4500 if kind == 'high' else 2500))
        if kind == 'high':
            return 3 - x
        elif kind == 'low':
            return -0.9 + x
        raise ValueError("kind must be high or low")

    class S2WidthHigh(Lichen):

        def pre(self, df):
            return S2Width.subpre(self, df)

        def _process(self, df):
            df.loc[:, self.name()] = (df.temp <= S2Width.relative_s2_width_bounds(df.s2, kind='high'))
            return df

    class S2WidthLow(RangeLichen):

        def pre(self, df):
            return S2Width.subpre(self, df)

        def _process(self, df):
            df.loc[:, self.name()] = (S2Width.relative_s2_width_bounds(df.s2, kind='low') <= df.temp)
            return df


S1AreaFractionTop = sciencerun0.S1AreaFractionTop

PreS2Junk = sciencerun0.PreS2Junk
