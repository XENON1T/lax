"""Cuts for post-SR1 analyses.

The SR1 cuts are copied, unless they are overriden below.
"""
import inspect

import numpy as np

import lax
from lax.lichen import ManyLichen, StringLichen  # pylint: disable=unused-import
from lax import __version__ as lax_version

from lax.lichens import sciencerun1 as sr1
DATA_DIR = sr1.DATA_DIR

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
    Wiki note: xenon:xenon1t:double_beta:cs2_aft_extension
    Contact: Dominick Cichon <dominick.cichon@mpi-hd.mpg.de>"""

    version = 1

    top_bound_string = ('(6.321778E-01 + 1.921483E-07 * cxys2 +'
                        ' -5.137170E-13 * cxys2**2 + 5.808641E-19 * cxys2**3 +'
                        ' -2.892108E-25 * cxys2**4 + 5.265449E-32 * cxys2**5 +'
                        ' 1.969023E+00 / sqrt(cxys2) + -4.036164E+00 / cxys2)')
    bot_bound_string = ('(6.190457E-01 + 3.533620E-08 * cxys2 +'
                        ' -1.692590E-13 * cxys2**2 + 2.106000E-19 * cxys2**3 +'
                        ' -1.179013E-25 * cxys2**4 + 2.459517E-32 * cxys2**5 +'
                        ' -1.421495E+00 / sqrt(cxys2) + -5.650175E+00 / cxys2)')

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
    for the old S2 desaturation algorithm (pax versions older than 6.10.0 ).

    Required minitrees: Corrections
    Defined with pax version: 6.8.0
    Wiki note: xenon:xenon1t:double_beta:cs2_aft_extension
    Contact: Dominick Cichon <dominick.cichon@mpi-hd.mpg.de>"""

    version = 0

    top_bound_string = ('(6.505760E-01 + 1.438548E-07 * cxys2 +'
                        ' -4.171372E-13 * cxys2**2 + 4.795966E-19 * cxys2**3 +'
                        ' -2.426551E-25 * cxys2**4 + 4.476144E-32 * cxys2**5 +'
                        ' 9.951226E-01 / sqrt(cxys2) + 5.901222E+00 / cxys2)')
    bot_bound_string = ('(6.184137E-01 + 3.521772E-09 * cxys2 +'
                        ' 2.802763E-14 * cxys2**2 + -1.134282E-19 * cxys2**3 +'
                        ' 7.049310E-26 * cxys2**4 + -1.243501E-32 * cxys2**5 +'
                        ' -1.388945E+00 / sqrt(cxys2) + -8.564284E+00 / cxys2)')

    string = ('((' + top_bound_string + ' > cs2_aft) & (' + bot_bound_string +
              ' < cs2_aft)) | (cxys2 > 2113500) | (cxys2 < 90)')

    def pre(self, df):
        df.loc[:, 'cxys2'] = ((df['cs2_top'] + df['cs2_bottom']) /
                                df['s2_lifetime_correction'])
        df.loc[:, 'cs2_aft'] = df['cs2_top'] / (df['cs2_top'] +
                                                df['cs2_bottom'])
        return df
