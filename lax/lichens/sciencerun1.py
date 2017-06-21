"""The cuts for science run 0

This includes all current definitions of the cuts for the first science run
"""

# -*- coding: utf-8 -*-
import inspect
import os
from pax import configuration

PAX_CONFIG = configuration.load_configuration('XENON1T')
from lax.lichen import ManyLichen, StringLichen
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
            FiducialCylinder1T(),
            InteractionExists(),
            S2Threshold(),
            InteractionPeaksBiggest(),
            S2AreaFractionTop(),
            S2SingleScatter(),
            DAQVeto(),
            S1SingleScatter(),
            S1AreaFractionTop(),
            S2PatternLikelihood(),
            S2Tails()
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
        self.lichen_list[5] = S2SingleScatterSimple()

        self.lichen_list += [
            S1PatternLikelihood(),
            S2Width(),
            S1MaxPMT()
        ]


class LowEnergyBackground(LowEnergyRn220):
    """Select background events with cs1<200

    This is the list that we'll use for the actual DM search. Additionally to the
    LowEnergyRn220 list it contains the PreS2Junk
    """

    def __init__(self):
        LowEnergyRn220.__init__(self)

        self.lichen_list += [
            PreS2Junk(),
        ]

class LowEnergyAmBe(LowEnergyRn220):
    """Select AmBe events with cs1<200 with appropriate cuts

    It is the same as the LowEnergyRn220 cuts, except uses an AmBe fiducial.
    """

    def __init__(self):
        LowEnergyRn220.__init__(self)

        # Replaces Fiducial
        self.lichen_list[0] = AmBeFiducial()

class LowEnergyNG(LowEnergyRn220):
    """Select AmBe events with cs1<200 with appropriate cuts

    It is the same as the LowEnergyRn220 cuts, except uses an AmBe fiducial.
    """

    def __init__(self):
        LowEnergyRn220.__init__(self)

        # Replaces Fiducial
        self.lichen_list[0] = NGFiducial()

        
DAQVeto = sciencerun0.DAQVeto

S2Tails = sciencerun0.S2Tails

FiducialCylinder1T = sciencerun0.FiducialCylinder1T

FiducialFourLeafClover1250kg = sciencerun0.FiducialFourLeafClover1250kg

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

InteractionPeaksBiggest = sciencerun0.InteractionPeaksBiggest

S1LowEnergyRange = sciencerun0.S1LowEnergyRange

S1MaxPMT = sciencerun0.S1MaxPMT

S1PatternLikelihood = sciencerun0.S1PatternLikelihood

S1SingleScatter = sciencerun0.S1SingleScatter

S2AreaFractionTop = sciencerun0.S2AreaFractionTop

S2SingleScatter = sciencerun0.S2SingleScatter

S2SingleScatterSimple = sciencerun0.S2SingleScatterSimple

S2PatternLikelihood = sciencerun0.S2PatternLikelihood

S2Threshold = sciencerun0.S2Threshold

S2Width = sciencerun0.S2Width

S1AreaFractionTop = sciencerun0.S1AreaFractionTop

PreS2Junk = sciencerun0.PreS2Junk
