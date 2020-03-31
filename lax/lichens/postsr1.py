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


class S1AreaFractionTop_he(Lichen):
    """Cut between  [0.05 - 99.9] percentile of the population in the parameter space Z vs S1AFT
    Note: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:arianna:s1_aft_highenergy
       Contact: arianna.rocchetti@physik.uni-freiburg.de 
       Cut defined above cs1>200 for the SingleScatter population. Valid also for Multiple scatter. 
       Requires: PatternReconstruction Minitree.
    """

    version = 2
    pars1 = [654.9, -754, 522.9, -151-8]  
    pars2 = [2548, -2182,  848.3, - 122]

    def cutline1(self, x):
        return self.pars1[0]*(x**3) + self.pars1[1] *(x**2) + self.pars1[2] * x + self.pars1[3]  

    def cutline2(self, x):
        return self.pars2[0]*(x**3) + self.pars2[1]  *(x**2) + self.pars2[2] * x + self.pars2[3]


    def _process(self, df):
        print("testing")
        df.loc[:, self.name()] = ( (df.z_3d_nn_tf>self.cutline1(df.s1_area_fraction_top) ) & 
                                (df.z_3d_nn_tf<self.cutline2(df.s1_area_fraction_top)  ) )
        return df
