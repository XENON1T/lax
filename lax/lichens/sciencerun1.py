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


class AmBeFiducial(StringLichen):
    """AmBe Fiducial volume cut.
    This uses the same Z cuts as the 1T fiducial cylinder,
    but a wider allowed range in R to maximize the number of nuclear recoils.
    There is a third cut on the distance to the source, so that we cut away background ER.
    Link to note:
    https://xecluster.lngs.infn.it/dokuwiki/lib/exe/fetch.php?media=xenon:xenon1t:hogenbirk:nr_band_sr0.html

    Contact: Erik Hogenbirk <ehogenbi@nikhef.nl>

    Position updated to reflect correct I-Belt 1 position. Link to Note:xenon:xenon1t:analysis:dominick:sr1_ambe_check
    """
    version = 2
    string = "(distance_to_source < 103.5) & (-92.9 < z) & (z < -9) & (sqrt(x*x + y*y) < 42.00)"

    def pre(self, df):
        source_position = (97, 43.5, -50)
        df.loc[:, 'distance_to_source'] = ((source_position[0] - df['x']) ** 2 +
                                           (source_position[1] - df['y']) ** 2 +
                                           (source_position[2] - df['z']) ** 2) ** 0.5
        return df


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

class S2Width(sciencerun0.S2Width):
    from scipy.stats import chi2
    version = 4
    diffusion_constant = 29.35 * ((units.cm)**2) / units.s
    v_drift = 1.335 * (units.um) / units.ns
    scg = 21.3 # s2_secondary_sc_gain in pax config
    scw = 229.58  # s2_secondary_sc_width median
    SigmaToR50 = 1.349

class S1SingleScatter(sciencerun0.S1SingleScatter):
    from scipy.stats import chi2
    version = 2
    s2width = S2Width

S1AreaFractionTop = sciencerun0.S1AreaFractionTop

PreS2Junk = sciencerun0.PreS2Junk
