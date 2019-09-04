"""Cuts for SR2 analyses"""
import numpy as np                                         # pylint: disable=unused-import
from lax.lichen import Lichen, ManyLichen, StringLichen    # pylint: disable=unused-import
from lax import __version__ as lax_version

from lax.lichens import sciencerun0 as sr0
from lax.lichens import sciencerun1 as sr1
from lax.lichens import postsr1
DATA_DIR = sr1.DATA_DIR

##
# Combination cut packages
##

class AllEnergy(ManyLichen):
    """Cuts applicable for low and high energy (gammas)

    This is a subset mostly of the low energy cuts.
    """
    version = lax_version

    def __init__(self):
        self.lichen_list = [
            S1PMT3fold(),
            FiducialZOptimized(),
            InteractionExists(),
            S2Threshold(),
            InteractionPeaksBiggest(),
            CS2AreaFractionTopExtended(),
            S2SingleScatter(),
            S2Width(),
            DAQVeto(),
            S1SingleScatter(),
            S2PatternLikelihood(),
            KryptonMisIdS1(),
            Flash(),
            PosDiff(),
            SingleElectronS2s()]


class LowEnergyRn220(AllEnergy):
    """Select Rn220 events with cs1<200

    This is the list that we use for the Rn220 data to calibrate ER in the
    region of interest.

    It doesn't contain the PreS2Junk cut
    """

    def __init__(self):
        AllEnergy.__init__(self)

        # Customize cuts for LowE data
        for idx, lichen in enumerate(self.lichen_list):

            # Replaces InteractionExists with energy cut (tighter)
            if lichen.name() == "CutInteractionExists":
                self.lichen_list[idx] = S1LowEnergyRange()

        # Add additional LowE cuts (that may not be tuned at HighE yet)
        self.lichen_list += [
            S1PatternLikelihood(),
            S1MaxPMT(),
            S1AreaFractionTop(),
            S1Width()
        ]

        # Add injection-position cuts (not for AmBe/NG)
        self.lichen_list += [
            S1AreaUpperInjectionFraction(),
            S1AreaLowerInjectionFraction()
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
            S2Tails(),  # Only for LowE background (#88)
            MuonVeto()
        ]


class LowEnergyAmBe(LowEnergyRn220):
    """Select AmBe events with cs1<200 with appropriate cuts

    It is the same as the LowEnergyRn220 cuts, except injection-related cuts
    """

    def __init__(self):
        LowEnergyRn220.__init__(self)

        # Remove cuts not applicable to AmBe/NG
        self.lichen_list = [lichen for lichen in self.lichen_list
                            if "InjectionFraction" not in lichen.name()]


class LowEnergyNG(LowEnergyAmBe):
    """Select NG events with cs1<200 with appropriate cuts

    It is the same as the LowEnergyAmBe cuts.
    """

    def __init__(self):
        LowEnergyAmBe.__init__(self)


##
# Livetime cuts
##

# DAQVeto
# Contact: Dan
DAQVeto = sr0.DAQVeto

# PMT Flash cut
# Contact: Oliver
Flash = sr0.Flash

# Muon Veto
# Contact: UNKNOWN!!
MuonVeto = sr0.MuonVeto

# S2Tails
# Contact: Fernando, Francesco
S2Tails = sr0.S2Tails


##
# S2 quality cuts
##

# S2 width
# Contact: Derek, LauraM
S2Width = sr1.S2Width

# Position reconstruction discrepancy cut
# Contact: UNKNOWN!!
PosDiff = sr0.PosDiff

# S2 AFT
# Contact: Giovanni, Dominick
class CS2AreaFractionTopExtended(StringLichen):
    """"An extension of CS2AreaFractionTop to the entire S2 range
    with a designed acceptance of 99%.
    It is defined in the (cxys2, cs2_aft) space, with:

    cxys2 = (cs2_top + cs2_bottom) / s2_lifetime_correction
    cs2_aft = cs2_top / (cs2_top + cs2_bottom)

    Events where cxys2 > 2163000 PE or cxys2 < 60 PE are not cut.

    This cut should be used with a pax version of at least 6.10.0
    due to a major update of the S2 desaturation correction.

    This is a preliminary version for SR2 analysis and should be used
    for runs after 18836, as only for them the SR2 S2 XY correction maps are
    valid. It is left to the discretion of the analyst whether they want to
    apply the cut also to other runs or not.

    Required minitrees: Corrections
    Defined with pax version: 6.10.1
    Wiki notes: xenon:xenon1t:double_beta:cs2_aft_extension
                xenon:xenon1t:double_beta:cs2_aft_extension_rejection_estimate
                xenon:xenon1t:analysis:sciencerun2:cs2_aft
    Contact: Dominick Cichon <dominick.cichon@mpi-hd.mpg.de>
             Giovanni Volta  <gvolta@physik.uzh.ch>"""

    version = 3

    top_bound_string = ('(6.533946E-01 + 2.238536E-07 * cxys2 +'
                        ' -5.791706E-13 * cxys2**2 + 6.021542E-19 * cxys2**3 +'
                        ' -2.786815E-25 * cxys2**4 + 4.763021E-32 * cxys2**5 +'
                        ' 2.021210E+00 / sqrt(cxys2) + -6.431296E+00 / cxys2)')
    bot_bound_string = ('(6.228859E-01 + 1.533585E-08 * cxys2 +'
                        ' -2.468763E-13 * cxys2**2 + 3.554834E-19 * cxys2**3 +'
                        ' -1.932773E-25 * cxys2**4 + 3.686730E-32 * cxys2**5 +'
                        ' -1.675048E+00 / sqrt(cxys2) + -1.291028E-01 / cxys2)')

    string = ('((' + top_bound_string + ' > cs2_aft) & (' + bot_bound_string +
              ' < cs2_aft)) | (cxys2 > 2163000) | (cxys2 < 60)')

    def pre(self, df):
        df.loc[:, 'cxys2'] = ((df['cs2_top'] + df['cs2_bottom']) /
                                df['s2_lifetime_correction'])
        df.loc[:, 'cs2_aft'] = df['cs2_top'] / (df['cs2_top'] +
                                                df['cs2_bottom'])
        return df

# S2PatternLikelihood
# Contact: Jingqiang
S2PatternLikelihood = postsr1.S2PatternLikelihood


##
# S1 quality cuts
##

# 3-fold coincidence PMT required for every S1
# Contact: Diego
class S1PMT3fold(Lichen):
    """
    The S1 PMT 3-fold cut rejects events in which, for the main S1, the number of PMTs
    with a hit close (window defined in pax) to the peak's sum waveform maximum
    is lower than 3.
    * Requires the 'Extended' minitrees (relies on 's1_tight_coincidence').
    * Retrieves the SR1/SR0 tight coincidence conditions.
    * Needs to be applied for all SR2 analyses (pax version newer than 6.8.0).
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:analysis:sciencerun2:3foldcut
    Contact: Diego Ramirez <diego.ramirez@physik.uni-freiburg.de>
    """

    version = 1

    def _process(self, df):
        df.loc[:, self.name()] = (df['s1_tight_coincidence'] > 2)
        return df

# Maximum contribution of each PMT to the S1 hitpattern
# Contact: UNKNOWN!!
S1MaxPMT = sr0.S1MaxPMT

# S1 AFT
# Contact: UNKNOWN!!
S1AreaFractionTop = sr0.S1AreaFractionTop

# S1 width
# Contact: UNKNOWN!!
S1Width = sr0.S1Width

# KryptonMisIdS1
# Contact: Evan
KryptonMisIdS1 = sr0.KryptonMisIdS1

# Remove S1s that are actually misidentified single electrons
# Contact: Fei
SingleElectronS2s = sr0.SingleElectronS2s

# S1 pattern likelihood
# Contact: Masatoshi
S1PatternLikelihood = sr0.S1PatternLikelihood

# Post-injection cuts for 220Rn calbration
# Contact: UNKNOWN!!
S1AreaUpperInjectionFraction = sr0.S1AreaUpperInjectionFraction
S1AreaLowerInjectionFraction = sr0.S1AreaLowerInjectionFraction


##
# Single scatter cuts
##

# S2 single scatter
# Contact: Tianyu, Yun
class S2SingleScatter(Lichen):
    """
    The single scatter is to cut an event if its largest other s2 have pattern goodness of fit better then
    normal electron pile up.
    * This is still preliminary.
    * To avoid cutting alpha events largest other s2 within 0 to 10e3 ns after s1 is exempt from the cut
    * Rely on InteractionPeaksBiggest to remove pile up event
    * Valid in S2 range 0-inf
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:sim:notes:tzhu:s2singlescattersr2
    Contact: Tianyu Zhu <tz2263@columbia.edu>
    """

    version = 5

    def _process(self, df):
        df.loc[:, self.name()] = True

        mask = df.eval('(largest_other_s2>0) \
            & (largest_other_s2_pattern_fit>0) \
            & ((largest_other_s2_delay_main_s1<0) \
            | (largest_other_s2_delay_main_s1>10e3))')

        df.loc[mask, self.name()] = df.loc[mask, 'largest_other_s2_pattern_fit'] > 0.856 * \
            df.loc[mask, 'largest_other_s2'] - 47.8 * \
            np.exp(- df.loc[mask, 'largest_other_s2'] / 32.93)

        return df

# S1 single scatter
# Contact: Joran, Jingqiang
S1SingleScatter = postsr1.S1SingleScatter


##
# Other cuts
##

# Fiducial volume
# Contact: UNKNOWN!!
FiducialZOptimized = sr1.FiducialZOptimized
FiducialCylinder1T = sr1.FiducialCylinder1T
FiducialCylinder1p3T = sr1.FiducialCylinder1p3T

# Interaction presence and dominance
# Contact: UNKNOWN!!
InteractionExists = sr0.InteractionExists
InteractionPeaksBiggest = sr0.InteractionPeaksBiggest

# PreS2Junk
# Contact: Florian
PreS2Junk = sr0.PreS2Junk

# cS1 in [0, 200]
# No contact. This is intended as a preselection.
S1LowEnergyRange = sr0.S1LowEnergyRange

# S2 < 200 PE
# Contact: UNKNOWN!!
S2Threshold = sr0.S2Threshold
