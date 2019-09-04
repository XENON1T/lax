"""Cuts for post-SR1 analyses.

The SR1 cuts are copied, unless they are overriden below.
"""
import inspect

import numpy as np

import lax
from lax.lichen import Lichen, ManyLichen, StringLichen  # pylint: disable=unused-import
from lax import __version__ as lax_version

from lax.lichens import sciencerun1 as sr1
DATA_DIR = sr1.DATA_DIR

from scipy import stats

# Import all lichens from sciencerun1
for x in dir(sr1):
    y = getattr(sr1, x)
    if inspect.isclass(y) and issubclass(y, lax.lichen.Lichen):
        locals()[x] = y

##
# Put new lichens here
##

class ERBandDEC(StringLichen):
    """A cut used in the double electron capture analysis to preselect the ER band
    Copied from xenon:xenon1t:analysis:backgrounds:ambe:fieguth:dec1t_final_analysis:cuts_overview

    Do NOT use at low energies (cS1 <200 PE)
    """
    version = 1
    string = "1 < log_cs_ratio < 2"

    def pre(self, df):
        df.loc[:, 'log_cs_ratio'] = np.log10(df['cs2']/df['cs1'])
        return df


class S2PatternLikelihood(StringLichen):
    """
    Extend S2 PatternLikelihood(S2 PLH) Cut up to 1.5e5 PE S2, which is good for up to around 220 keVee. This cut is a
    combination of SR1 S2PLH(s2 < 10000 PE) and extension of S2 PLH(1e4 < S2 < 1.5e5 PE), thus have same performance as 
    sr1 S2 PLH for low energy and have improved performance for high energy. S2 PLH cut aims to remove poorly 
    reconstructed events.

    The details can be found below:
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:jingqiang:darkphoton:s2patternlikelihood_200kev

    Requires Extended minitrees.
    Contact: Jingqiang Ye <jiy171@ucsd.edu>
    """
    version = 0
    p0 = (0.072, 594)
    p1_sr1 = (0.0404, 594, 0.0737, -686)


    string = ("((s2_pattern_fit < %.3f*s2 + %.3f) & (s2 > 10000))\
             | ((s2_pattern_fit < %.3f*s2 + %.3f*s2**%.3f + %.3f) & (s2 < 10000))" %(p0 + p1_sr1))


class CS2AreaFractionTopExtended(StringLichen):
    """"An extension of CS2AreaFractionTop to the entire S2 range
    with a designed acceptance of 99%.
    It is defined in the (cxys2, cs2_aft) space, with:

    cxys2 = (cs2_top + cs2_bottom) / s2_lifetime_correction
    cs2_aft = cs2_top / (cs2_top + cs2_bottom)

    Events where cxys2 > 2012700 PE or cxys2 < 90 PE are not cut.

    This cut should be used with a pax version of at least 6.10.0
    due to a major update of the S2 desaturation correction.

    Required minitrees: Corrections
    Defined with pax version: 6.10.1
    Wiki notes: xenon:xenon1t:double_beta:cs2_aft_extension
                xenon:xenon1t:double_beta:cs2_aft_extension_rejection_estimate
    Contact: Dominick Cichon <dominick.cichon@mpi-hd.mpg.de>"""

    version = 2

    top_bound_string = ('(6.306642E-01 + 2.032259E-07 * cxys2 +'
                        ' -5.450833E-13 * cxys2**2 + 6.182750E-19 * cxys2**3 +'
                        ' -3.087908E-25 * cxys2**4 + 5.637659E-32 * cxys2**5 +'
                        ' 2.123849E+00 / sqrt(cxys2) + -4.258579E+00 / cxys2)')
    bot_bound_string = ('(6.190286E-01 + 3.546500E-08 * cxys2 +'
                        ' -1.696261E-13 * cxys2**2 + 2.110380E-19 * cxys2**3 +'
                        ' -1.181304E-25 * cxys2**4 + 2.463864E-32 * cxys2**5 +'
                        ' -1.419829E+00 / sqrt(cxys2) + -5.642919E+00 / cxys2)')

    string = ('((' + top_bound_string + ' > cs2_aft) & (' + bot_bound_string +
              ' < cs2_aft)) | (cxys2 > 2012700) | (cxys2 < 90)')

    def pre(self, df):
        df.loc[:, 'cxys2'] = ((df['cs2_top'] + df['cs2_bottom']) /
                                df['s2_lifetime_correction'])
        df.loc[:, 'cs2_aft'] = df['cs2_top'] / (df['cs2_top'] +
                                                df['cs2_bottom'])
        return df


class CS2AreaFractionTopExtendedOldDesat(StringLichen):
    """"An extension of CS2AreaFractionTop to the entire S2 range
    with a designed acceptance of 99%.
    It is defined in the (cxys2, cs2_aft) space, with:

    cxys2 = (cs2_top + cs2_bottom) / s2_lifetime_correction
    cs2_aft = cs2_top / (cs2_top + cs2_bottom)

    Events where cxys2 > 2113500 PE or cxys2 < 90 PE are not cut.

    This is an alternate version of CS2AreaFractionTopExtended
    for the old S2 desaturation algorithm (pax versions older than 6.10.0).

    Required minitrees: Corrections
    Defined with pax version: 6.8.0
    Wiki notes: xenon:xenon1t:double_beta:cs2_aft_extension
                xenon:xenon1t:double_beta:cs2_aft_extension_rejection_estimate
    Contact: Dominick Cichon <dominick.cichon@mpi-hd.mpg.de>"""

    version = 1

    top_bound_string = ('(6.499452E-01 + 1.473286E-07 * cxys2 +'
                        ' -4.273597E-13 * cxys2**2 + 4.922129E-19 * cxys2**3 +'
                        ' -2.493411E-25 * cxys2**4 + 4.603164E-32 * cxys2**5 +'
                        ' 1.065584E+00 / sqrt(cxys2) + 5.840339E+00 / cxys2)')
    bot_bound_string = ('(6.184080E-01 + 3.561342E-09 * cxys2 +'
                        ' 2.792023E-14 * cxys2**2 + -1.133069E-19 * cxys2**3 +'
                        ' 7.043299E-26 * cxys2**4 + -1.242414E-32 * cxys2**5 +'
                        ' -1.388373E+00 / sqrt(cxys2) + -8.562613E+00 / cxys2)')

    string = ('((' + top_bound_string + ' > cs2_aft) & (' + bot_bound_string +
              ' < cs2_aft)) | (cxys2 > 2113500) | (cxys2 < 90)')

    def pre(self, df):
        df.loc[:, 'cxys2'] = ((df['cs2_top'] + df['cs2_bottom']) /
                                df['s2_lifetime_correction'])
        df.loc[:, 'cs2_aft'] = df['cs2_top'] / (df['cs2_top'] +
                                                df['cs2_bottom'])
        return df


class MisIdS1SingleScatter(Lichen):
    """Cut to target the shoulder on Kr83m data due to mis-identified krypton events.

    This cut is defined in the space of cs1 and largest_s2_before_main_s2_area.
    It's possible for one of the conversion electrons of the Kr83m decay to be mis-classified as an S2, 
    leading to an S2 that occurs in time before the main S2 that is larger than typical single-electrons. 
    Requires Extended minitrees.

    Note: xenon:shockley:misids1singlescatter
    Contact: Evan Shockley <ershockley@uchicago.edu>
    """

    version = 1.1
    
    pars = [60, 1.04, 4]  # from a fit to target only mis-Id Kr83m events
    s1_thresh = 155  # cs1 PE. Up to this value the cut will be a straight line
    cutval = 125 # straight line part of the cut
    min_s1 = 25 # smallest S1 to consider cutting
    
    
    def _cutline(self, x):
        return self.pars[0] + (self.pars[1] / 1e6) * (x - 220) ** self.pars[2]
    
    def cutline(self, x):
        return np.nan_to_num(self.cutval * (x < self.s1_thresh)) + np.nan_to_num((x >= self.s1_thresh) * self._cutline(x))
    
    def _process(self, df):
        df.loc[:, self.name()] = (np.nan_to_num(df.largest_s2_before_main_s2_area) < self.cutline(df.cs1)) | \
                                 (df.cs1 < self.min_s1)
        return df


class S1SingleScatter(Lichen):
    """
    Added more requirement for largest_other_s1 to be identified as problematic. We found many largest_other_s1 are
    from AP after S1, thus we required the largest hit area in one PMT shouldn't not exceed certain amount of the
    largest_other_s1. The function is a bit different from S1 MAX PMT, as the current one seems not strict at
    very low energy (~5 PE S1).

    Requires Extended minitrees.

    The cut is investigated for both SR1 and SR2 here:
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:low_energy_er:s1singlescattercut

    It should be applicable to data regardless whether it is ER or NR.
    Contacts:
    Jacques Pienaar, <jpienaar@uchicago.edu>
    Joran Angevaare, <j.angevaare@nikhef.nl>
    Jingqiang Ye, <jiy171@ucsd.edu>
    """

    version = 6
    s2width = S2Width
    alt_s1_coincidence_threshold = 3

    @staticmethod
    def largest_area_threshold(s1):
        """
        threshold of largest hit area in a single PMT for largest_other_s1, similar to S1 MAX PMT cut.
        """
        return np.minimum(0.052 * s1 + 4.15, 0.6 * s1 - 0.5)

    def _process(self, df):
        df.loc[:, self.name()] = True  # Default is True
        mask = (df.alt_s1_interaction_drift_time > self.s2width.DriftTimeFromGate) & (
                df.alt_s1_tight_coincidence >= self.alt_s1_coincidence_threshold)

        # S2 width cut for alternate S1 - main S2 interaction
        alt_n_electron = np.clip(df.loc[mask, 's2'], 0, 5000) / self.s2width.scg

        alt_rel_width = np.square(df.loc[mask,
                                         's2_range_50p_area'] / self.s2width.SigmaToR50) - np.square(self.s2width.scw)
        alt_rel_width /= np.square(self.s2width.s2_width_model(self.s2width,
                                                               df.loc[mask, 'alt_s1_interaction_drift_time']))

        alt_interaction_passes = stats.chi2.logpdf(
            alt_rel_width * (alt_n_electron - 1), alt_n_electron) > - 20

        alt_pmt_passes = df.loc[mask, 'alt_s1_largest_hit_area'] < \
                         self.largest_area_threshold(df.loc[mask, 'largest_other_s1'])

        df.loc[mask, (self.name())] = True ^ (alt_interaction_passes & alt_pmt_passes)

        return df