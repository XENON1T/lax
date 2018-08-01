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


class CS2AreaFractionTopExtended(StringLichen):
    """"An extension of CS2AreaFractionTop to the entire S2 range
    with a designed acceptance of 99%.
    It is defined in the (cxys2, cs2_aft) space, with:

    cxys2 = (cs2_top + cs2_bottom) / s2_lifetime_correction
    cs2_aft = cs2_top / (cs2_top + cs2_bottom)

    Events where cxys2 > 1922700 PE are not cut.

    Required minitrees: Corrections
    Defined with pax version: 6.8.0
    Wiki note: xenon:xenon1t:double_beta:cs2_aft_extension
    Contact: Dominick Cichon <dominick.cichon@mpi-hd.mpg.de>"""

    version = 0

    top_bound_string = ('(6.480304E-01 + 1.445442E-07 * cxys2 +'
                        ' -4.443394E-13 * cxys2**2 + 5.440908E-19 * cxys2**3 +'
                        ' -2.925384E-25 * cxys2**4 + 5.716877E-32 * cxys2**5 +'
                        ' 1.060000E+00 / sqrt(cxys2) + 5.297425E+00 / cxys2)'
    bot_bound_string = ('(6.135743E-01 + 3.979536E-08 * cxys2 +'
                        ' -8.288567E-14 * cxys2**2 + 2.617481E-20 * cxys2**3 +'
                        ' -5.108758E-27 * cxys2**4 + 2.312916E-33 * cxys2**5 +'
                        ' -6.921489E-01 / sqrt(cxys2) + -2.507866E+01 / cxys2)'

    string = ('((' + top_bound_string + ' > cs2_aft) & (' + bot_bound_string +
              ' < cs2_aft)) | cxys2 > 1922700')

    def pre(self, df):
        df.loc[:, 'cxys2'] = ((df['cs2_top'] + df['cs2_bottom']) /
                                df['s2_lifetime_correction'])
        df.loc[:, 'cs2_aft'] = df['cs2_top'] / (df['cs2_top'] +
                                                df['cs2_bottom'])
        return df
