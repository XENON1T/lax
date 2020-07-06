"""Cuts for post-SR1 analyses.

The SR1 cuts are copied, unless they are overriden below.
"""
import inspect

import numpy as np
import pandas as pd
import pickle
import os

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


class Volume_DBD(StringLichen):
    """
    Note: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:chiara:fv:binned#update_july_6_2020
    
    Fiducial volume defined for 0vbb decay search: data outside the blinded region, from the Bi214 peak at 2.2 MeV and the Tl208 at 2.6 MeV are used.
    The TPC volume is binned and a figure of merit, from the counts of the background, is computed in each bin. In this way we get a volume distribution of the sensitivity in the volume.
    From here it's possible to define a fiducial volume whose shape is not symmetric in the TPC.
    The contours of the figure of merit in the volume are fit with two semiellipsiod functions, for the upper and lower boundaries.
    The lower limit is set to not go below -94 cm. 
    The figure of merit is recalculated in the volumes and the maximum is found for a fiducial mass of ~741 kg.
    
    Contact: Chiara Capelli <chiara@physik.uzh.ch>
    """
    
    bottom_tpc = -94

    def SuperEllipseUpperZs(x, zloc, zscale, r2scale, power_const):
        Zs = np.power(
            1. - np.power(
                x/r2scale,
                power_const
            ),
            1./power_const
        )*zscale+zloc           
        return Zs

    def SuperEllipseLowerZs(x, zloc, zscale, r2scale, power_const):
        Zs = -np.power(
            1. - np.power(
                x/r2scale,
                power_const
            ),
            1./power_const
        )*zscale+zloc
        Zs = np.array(Zs)    
        return np.where((Zs<bottom_tpc),bottom_tpc,Zs)
    
    par_up = [-213.6148297920783, 188.07255694010047, 1347.1771882218404, 6.491616457942884]
    par_low = [-7.955115883403868, 86.62520910736373, 1317.2991340021406, 8.338709398342486]
        
    def _process(self, df):
        df.loc[:, self.name()] = (df.z_3d_nn_tf > SuperEllipseLowerZs(df.r_3d_nn_tf**2,par_low[0],par_low[1],par_low[2],par_low[3]) 
                                  & df.z_3d_nn_tf < SuperEllipseUpperZs(df.r_3d_nn_tf**2,par_up[0],par_up[1],par_up[2],par_up[3]))
            
        return df


class ERband_HE(StringLichen):
    """"ERband cut at 1-99 percentiles, tuned on SR1 background data. 
    It is defined in the (Log10(cs2bottom/cs1) vs ces) space, with:

    ces = cs1/g1[z]) + cs2_bottom/g2[z]*w
    z = z_3d_nn_tf
    g1(z) = 0.14798+(0.00007*z)
    g2(z) = 10.504-(0.015*z)
    w=13.7e-3
 
    It returns the df after the ERband_HE cut, including a new variable called 'ces_ERband_HE'.
    The ERband definition uses data between 50—3020 keV with a precut z>-50.
    Below 50 keV events will pass the cut. From 2 MeV to infinity (excluding the the blind region) the ERband is defined as the  average 1st and 99th percentile between 2-2.4 MeV. In the bling region the ER band is the average of the band defined between 1.190-1.220 MeV.  
    The text file with the cut (ces value, Q50, Q99, Q1) can be found in /dali/lgrandi/manenti/cuts/ERband_HE/
    and in ../data/. 
    
    Required minitrees: Corrections
    Defined with pax version: 6.10.1
    Wiki notes: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:manenti:sr1_erband_v0bb
    Contact: Laura Manenti <laura.manenti@nyu.edu>"""

    version = 1  

    def _process(self, df):
        
        #load mean, sigma values
        ERband = np.genfromtxt('/dali/lgrandi/manenti/cuts/ERband_HE/ERband_Q50_Q99_Q1_50toInf_gapAs1.2MeV.txt',skip_header=1)
        Q99 = ERband[:,2]
        Q1 = ERband[:,3]
        
        #array bins
        ces_bin = ERband[:,0]
        
        #define ces
        w=13.7e-3
        df.loc[:, 'ces_ERband_HE'] = w*(df.cs1_nn_tf/self.g1_sr1_he_ap(df.z_3d_nn_tf) +
                                  df.cs2_bottom_nn_tf/self.g2_sr1_he_ap(df.z_3d_nn_tf))
        x = df['ces_ERband_HE'] 
        inds = np.digitize(x, ces_bin) #indices of the bins to which each value in x belongs. 
        
        #get corresponding cut values
        cut_top = [Q99[i-1] for i in inds]
        cut_bottom = [Q1[i-1] for i in inds]
        
        df.loc[:, self.name()] = True # default is True 
        df.loc[:, self.name()] = ( ( np.log10(df['cs2_bottom']/df['cs1']) < cut_top ) \
                                       & ( np.log10(df['cs2_bottom']/df['cs1']) > cut_bottom ) )       
        return df

    def g1_sr1_he_ap(self, z):
        return 0.14798+(0.00007*z)

    def g2_sr1_he_ap(self, z):
        return 10.504-(0.015*z)


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
    Extend S2PatternLikelihood cut at High Energy (from 0 to 3000 keVee). 
    This cut is a combinaison of the low energy (s2 < 1e4 PE - develop for SR1) and the one extended at high energy.
    
    Details can be found in this note:
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:chloetherreau:0vbb_s2_likelihood_cut_he_update
    
    Long to process, applied it after all other cuts
    Requires S2PatternReducedAP minitrees (hax PR:https://github.com/XENON1T/hax/pull/259)
    Contact: Chloe Therreau <chloe.therreau@subatech.in2p3.fr>
    """
    def pre(self, df):
        def powerlaw(x,amp0,power0,amp1,power1,cte):
            return amp0*x**power0+amp1*x**power1+cte   

        phi1=4
        phi2=10
        phi3=16
        phi4=22

        # Load parameters
        params_load = np.loadtxt('/dali/lgrandi/ctherreau/cuts/S2PatternHE/s2patternlikelihoodcut_he_r_phi_params_v2.txt')
        # Reshape parameters
        params=[params_load[:phi1], params_load[phi1:phi1+phi2],
                params_load[phi1+phi2:phi1+phi2+phi3],params_load[phi1+phi2+phi3:phi1+phi2+phi3+phi4]]
        
        r_here = 'r_3d_nn_tf'
        phi_here = 'phi_3d_nn_tf'
        df.loc[:,phi_here] = np.arccos(df.x_3d_nn_tf/df.r_3d_nn_tf)*np.sign(df.y_3d_nn_tf)
        df_list=[]
        R = np.linspace(0, 47, 5) 
        for i in range(len(R)-1):
            n = [phi1,phi2,phi3,phi4][i]
            td = 2*np.pi/n
            for j in range(n):
                tmin, tmax, rmin, rmax = j*td-np.pi, j*td-np.pi+td, R[i], R[i+1]
                df_box_cut = df.copy()
                box_cut = ((df_box_cut[r_here]>rmin)&(df_box_cut[r_here]<rmax)
                         &(df_box_cut[phi_here]>tmin)&(df_box_cut[phi_here]<tmax))
                df_box_cut = df_box_cut[box_cut]
                a_here = params[i][j][0]*np.ones(len(df[box_cut]))
                b_here = params[i][j][1]*np.ones(len(df[box_cut]))
                c_here = params[i][j][2]*np.ones(len(df[box_cut]))
                d_here = params[i][j][3]*np.ones(len(df[box_cut]))
                e_here = params[i][j][4]*np.ones(len(df[box_cut]))
                df_box_cut.loc[:,'CutS2PatternLikelihoodHE_a'] = a_here
                df_box_cut.loc[:,'CutS2PatternLikelihoodHE_b'] = b_here
                df_box_cut.loc[:,'CutS2PatternLikelihoodHE_c'] = c_here
                df_box_cut.loc[:,'CutS2PatternLikelihoodHE_d'] = d_here
                df_box_cut.loc[:,'CutS2PatternLikelihoodHE_e'] = e_here

                df_list.append(df_box_cut)
                del df_box_cut
        del df
        df=pd.concat(df_list)
        df.loc[:,'log10_s2_pattern_fit_top_reduced_ap']=np.log10(df['s2_pattern_fit_top_reduced_ap'])
        df.loc[:,'log10_s2']=np.log10(df['s2'])

        return df
    p0 = (0.072, 594)
    p1_sr1 = (0.0404, 594, 0.0737, -686)
    
    string = (" ((log10_s2_pattern_fit_top_reduced_ap<\
                    CutS2PatternLikelihoodHE_a*log10_s2**CutS2PatternLikelihoodHE_b+\
                    CutS2PatternLikelihoodHE_c*log10_s2**CutS2PatternLikelihoodHE_d+\
                    CutS2PatternLikelihoodHE_e) & (s2 > 1e4))  \
              | ((s2_pattern_fit < %.3f*s2 + %.3f*s2**%.3f + %.3f) & (s2 < 10000) )"%(p1_sr1))

class CS2AreaFractionTopExtended(Lichen):
    """"An extension of CS2AreaFractionTop to the entire S2 range
    with a designed acceptance of 99%.
    It is defined in the (s2_no_ap_pmts, cs2_aft_no_ap_pmts) space
    and valid for events with r_3d_nn < 36.94.
    Both cut space parameters are from the S2WithoutAfterpulsePMTs
    minitree.

    Events with s2_no_ap_pmts > 1427769.230769 PE or
    s2_no_ap_pmts < 71.428571 PE are not cut.

    This cut should be used with a pax version of at least 6.10.0
    due to a major update of the S2 desaturation correction compared
    to previous versions. Acceptance is most homogenous in XY for
    events with z_3d_nn < -25 or s2 < 2e5.

    Required minitrees: S2WithoutAfterpulsePMTs
    Defined with pax version: 6.10.1
    Wiki note: xenon:xenon1t:double_beta:cs2_aft_no_ap_pmts
    Contact: Dominick Cichon <dominick.cichon@mpi-hd.mpg.de>"""

    version = 3

    # define cut line function
    top_params = [-2.754897E+06, -1.579777E+06, 1.475401E-05, 4.299098E-32,
                  -1.622189E-25, 2.303079E-19, -1.584616E-13, 5.977093E-08,
                  6.275617E-01]
    bot_params = [3.172141E+00, -1.776482E+00, -9.837439E-05, 5.693593E-32,
                  -2.557269E-25, 4.323291E-19, -3.599242E-13, 2.158422E-07,
                  6.312527E-01]

    def inv_sqrt(self, x, p0, p1):
        return p0 / (1 + p1 * x**(1/2.))

    def aft_cut_line(self, x, *args):
        return self.inv_sqrt(x, args[0], args[1]) + args[2] * np.sqrt(x) + np.polyval(args[3:], x)

    def _process(self, df):
        df.loc[:, self.name()] = (((df.cs2_aft_no_ap_pmts < self.aft_cut_line(df.s2_no_ap_pmts, *self.top_params)) &
                                   (df.cs2_aft_no_ap_pmts > self.aft_cut_line(df.s2_no_ap_pmts, *self.bot_params))) |
                                  (df.s2_no_ap_pmts > 1427769.230769) |
                                  (df.s2_no_ap_pmts < 71.428571))

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
    """
    Cut between  [0.05 - 99.9] percentile of the population in the parameter space Z vs S1AFT

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
        df.loc[:, self.name()] = ( (df.z_3d_nn_tf>self.cutline1(df.s1_area_fraction_top) ) & 
                                (df.z_3d_nn_tf<self.cutline2(df.s1_area_fraction_top)  ) )
        return df
   
class PosDiff_HE(Lichen):
    """
    Note: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:chiara:posdiffcut_update
    This cut is defined for removing the events with large position difference between NN_TF and TPF algorithm.
    Defined for high energy analysis and whole background matching up to 3 MeV.
    Contact: Chiara Capelli <chiara@physik.uzh.ch>
    """
    version = 0.1
    
    def _process(self, df):
        df.loc[:, self.name()] = np.sqrt((df['x_observed_nn_tf']-df['x_observed_tpf'])**2+
                                         (df['y_observed_nn_tf']-df['y_observed_tpf'])**2)<(3569.674 * np.exp(-np.log10(df.s2)/0.369) + 1.582)
        return df


class S2SingleScatter_HE(Lichen):
    """This cut is for cutting multiple s2 events for high energy er events, s2>1e3
    See note xenon:xenon1t:sim:notes:tzhu:s2singlescatterpostsr1
    Contact: Tianyu Zhu <tz2263@columbia.edu>
    """
    version = 0.1
    gmix_filename = os.path.join(DATA_DIR, 's2_single_classifier_gmix_v6.10.0.pkl')
    gmix = pickle.load(open(gmix_filename, 'rb'))

    def _process(self, df):
        df[self.name()] = True
        mask = np.logical_and(df['largest_other_s2_pattern_fit']>0, df['s2']>0)
        Y = np.log10(df.loc[mask, ['largest_other_s2', 'largest_other_s2_pattern_fit', 's2']])
        df.loc[mask, self.name()] = self.gmix.predict(Y).astype(bool)
        return df


class S2Width_HE(Lichen):
    """
    Note: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:wolf:s2width_dbd_151119

    This cut is defined for removing the events with an unphysical S2Width at high energies.
    Defined for high energy analysis and MC matching covering the whole background starting
    from 0 keV up to 3 MeV.
    At low energies (CES < 250keV) the standard S2Width cut is used. Above 250keV the cut based on
    a background sample and quantiles at 1% and 99% is derived. The cut space in which the quantiles
    are derived is s2_range_50p_area vs drift_time. The cut values are derived in a sample with
    CES > 200keV and CES < 3000keV.

    Required minitrees: Corrections
    Defined with pax version: 6.10.1

    Contact: Chiara Capelli (chiara@physik.uzh.ch)
    Tim Michael Heinz Wolf (tim.wolf@mpi-hd.mpg.de)
    """
    version = 0.1

    def _process(self, df):
        # load cut values
        cut_array = np.loadtxt("/project2/lgrandi/twolf/S2WidthCutFiles/cut_values.txt")
        drift_time_bin_centers = (cut_array[:, 0])
        drift_time_edges = drift_time_bin_centers + 5 # total bin width is 10

        # find bin in cutspace for drift_time
        found_bin = np.digitize(df["drift_time"], drift_time_edges)

        # apply standard S2 width cut
        S2WidthLichen = sr1.S2Width()
        df = S2WidthLichen.process(df)

        mybins = []
        for bin in found_bin:
            if bin == len(drift_time_edges):
                mybins.append(bin - 1)
            else:
                mybins.append(bin)
        found_bin = np.asarray(mybins)

        # get corresponding cut values
        cut_down = cut_array[found_bin][:, 1]
        cut_up = cut_array[found_bin][:, 2]

        # derivation of combined energy to stich the two cuts together
        w=13.7e-3
        df_test = pd.DataFrame()
        df_test.loc[:, 'CES'] = w*(df.cs1_nn_tf/self.g1_sr1_he_ap(df.z_3d_nn_tf) +
                                  df.cs2_bottom_nn_tf/self.g2_sr1_he_ap(df.z_3d_nn_tf))
        df_test.loc[:, self.name()] = ((df["s2_range_50p_area"] > cut_down) & (df["s2_range_50p_area"] < cut_up))

        # stiching the cuts together
        df.loc[:, self.name()] = True # default is True
        df.loc[:, self.name()] = np.where(df_test['CES'] < 250, df['CutS2Width'], df_test[self.name()])
        return df

    def g1_sr1_he_ap(self, z):
        return 0.14798+(0.00007*z)

    def g2_sr1_he_ap(self, z):
        return 10.504-(0.015*z)


class S1SingleScatter_HE(Lichen):
    """
    Note: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:chloetherreau:he_s1_single_scatter_cut
    This cut is defined for selecting events with only one S1 in the waveform
    Contact: Chloe Therreau (chloe.therreau@subatech.in2p3.fr)
             Tim Michael Heinz Wolf (tim.wolf@mpi-hd.mpg.de)
    """
    version = 0.1
    
    def _process(self, df):
        df.loc[:, self.name()] = df['largest_other_s1']<45
        return df
