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
    combination of SR1 S2PLH(s2 < 10000 PE) and extension of S2 PLH(1e4 < S2 < 1.5e5 PE), thus have same performance as sr1 S2 PLH for low energy and have improved performance for high energy. S2 PLH cut aims to remove poorly reconstructed events.

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
