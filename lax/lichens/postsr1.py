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


class MisIdS1SingleScatter(Lichen):
    """
    Removes events that should be cut by the other single scatter cuts, but are not because of mis-classified S1s.
    This was tuned specifically for the Kr83m peak at 32 keV, which remains after cuts because the 9 keV S1 was
    classfied as an S2.
    Required treemakers: Extended, LargestPeakProperties, Corrections
    """

    version = 1.0
    # only consider events where the s1 is between the following
    s1_area_cut = (0, 500)
    # only consider suspicious s2s above this area
    area_cut = 60
    # only consider suspcious s2s with time differences (main_s2_time - this_s2_time) greater than this
    min_dt = S2Width.DriftTimeFromGate
    # only consider events where the suspicious s2 is this close to the main s1 (ns)
    s1_timediff_cut = 10000
    # chi2 value to cut on to see if the paired (misID?) S2 and main S2 give a valid width
    chi2_cut = -25
    
    def pre(self, df):
        # time difference between suspicious s2s and main s2
        df['suspicious_s2_1_drift_time'] = df.s2_center_time - df.largest_s2_before_main_s2_time
        df['suspicious_s2_1_delay_main_s1'] = df.s1_center_time - df.largest_s2_before_main_s2_time
        df['suspicious_s2_2_drift_time'] = df.s2_center_time - df.secondlargest_s2_before_main_s2_time
        df['suspicious_s2_2_delay_main_s1'] = df.s1_center_time - df.secondlargest_s2_before_main_s2_time
        # variable used in width model
        df['suspicious_s2_drift_time'] = float('nan')
        return df

    def _process(self, df):
        df.loc[:, self.name()] = True  # Default is True
        
        # define an s1 area cut because large s1s produce photoionization
        # 600 was chosen because it includes all of Kr83m spectrum
        mask1 = (self.s1_area_cut[0]<df.cs1) & (df.cs1 < self.s1_area_cut[1])
        # look for s2s before the main s2 (> some value to exclude SEs)
        mask2a = df.largest_s2_before_main_s2_area > self.area_cut
        mask2b = df.secondlargest_s2_before_main_s2_area > self.area_cut
        # the time between the suspect s2 and main s2 should be > gate drift time
        mask3a = df.suspicious_s2_1_drift_time > self.min_dt
        mask3b = df.suspicious_s2_2_drift_time > self.min_dt
        # and since we have an S2Width cut, the suspicious S2 should be pretty close to the main S1
        mask4a = np.absolute(df.suspicious_s2_1_delay_main_s1 < self.s1_timediff_cut)
        mask4b = np.absolute(df.suspicious_s2_2_delay_main_s1 < self.s1_timediff_cut)
        
        
        # combine the masks
        maska = (mask2a & mask3a & mask4a)
        maskb = (mask2b & mask3b & mask4b)
        mask = mask1 & ( maska | maskb)
        
        # if the secondlargest s2 looks suspicious, use that drift time in width model
        df.loc[maskb, 'suspicious_s2_drift_time'] = df['suspicious_s2_2_drift_time']
        # if largest s2 looks suspicious, use that time. This overrides the previous line so we use biggest
        df.loc[maska, 'suspicious_s2_drift_time'] = df['suspicious_s2_1_drift_time']
        
        
        # check if the suspect s2 is consistent with s2width model (copy+pasted from S2Width cut)
        alt_n_electron = np.clip(df.loc[mask, 's2'], 0, 5000) / S2Width.scg
        alt_rel_width = np.square(df.loc[mask, 's2_range_50p_area'] / S2Width.SigmaToR50) - \
                        np.square(S2Width.scw)
        compare_widths = alt_rel_width / np.square(S2Width.s2_width_model(S2Width,
                                                                          df.loc[mask, 'suspicious_s2_drift_time']))
        chi2s = stats.chi2.logpdf(compare_widths * (alt_n_electron - 1), alt_n_electron)
        
        # now check chi2
        alt_interaction_passes = chi2s > self.chi2_cut
        df.loc[mask, self.name()] = True ^ alt_interaction_passes
        return df
