# -*- coding: utf-8 -*-
import numpy as np
from pax import units, configuration

PAX_CONFIG = configuration.load_configuration('XENON1T')
from lax.lichen import Lichen, RangeLichen, ManyLichen
from lax import __version__ as lax_version


class AllCuts(ManyLichen):
    version = lax_version

    def __init__(self):
        self.lichen_list = [
            FiducialCylinder1T(),
            InteractionExists(),
            S2Threshold(),
            InteractionPeaksBiggest(),
            S2AreaFractionTop(),
            S2SingleScatter(),
            DAQVetoCut(),
        ]


class LowEnergyCuts(AllCuts):
    def __init__(self):
        AllCuts.__init__(self)
        self.lichen_list[1] = S1LowEnergyRange()        
        
        self.lichen_list += [
            S1PatternLikelihood(),
            S2Width(),
            S1MaxPMT(),
        ]
    

class InteractionExists(RangeLichen):
    """Checks that there was a pairing of S1 and S2.
        
    """
    version = 0
    allowed_range = (0, np.inf)
    variable = 'cs1'


class S2Threshold(RangeLichen):
    """The S2 energy at which the trigger is perfectly efficient.

    See: https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:firstresults:daqtriggerpaxefficiency
    """
    version = 0
    allowed_range = (150, np.inf)
    variable = 's2'


class S1LowEnergyRange(RangeLichen):
    """For isolating the low-energy band.
    
    """
    version = 0
    allowed_range = (0, 200)
    variable = 'cs1'

class S1MaxPMT(Lichen):
    """Cut events which have a high fraction of the area in a single PMT
    
    Cuts events which are mostly seen by one PMT.
    These events could be for example afterpulses or light emission. 
    This is the 99% quantile fit using pax 6.4.2 on Rn220 
    
    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:yuehuan:analysis:0sciencerun_s1_pmtmax
    
    Author: Julien Wulf <jwulf@physik.uzh.ch>
    """
    def pre(self, df):
        df.loc[:,'temp'] = 0.052 * df['s1'] + 4.15

    def _process(self, df):
        df.loc[:, self.__class__.__name__] = df['largest_hit_channel'] < df.temp
        return df
    
class FiducialCylinder1T(ManyLichen):
    """Fiducial volume cut.

    The fidicual volume cut defines the region in depth and radius that we
    trust and use for the exposure. This is the region where the background
    distribution is flat.

    This version of the cut is based pax v6.2 bg run 0 data. See the
    note first results fiducial volume note for the study of the definition.

    Author: Sander breur sanderb@nikhef.nl

    """
    version = 2

    def __init__(self):
        self.lichen_list = [self.Z(),
                            self.R()]

    class Z(RangeLichen):
        allowed_range = (-83.45, -13.45)
        variable = 'z'

    class R(RangeLichen):
        variable = 'r'  #  Should add to minitrees

        def pre(self, df):
            df.loc[:, self.variable] = np.sqrt(df['x'] * df['x'] + df['y'] * df['y'])
            return df

        allowed_range = (0, 39.85)


class S2AreaFractionTop(RangeLichen):
    """Cuts events with an unusual fraction of S2 on top array.

    Primarily cuts gas events with a particularly large S2 AFT, also targets some
    strange / junk / other events with a low AFT.

    Author: Adam Brown abrown@physik.uzh.ch
    """
    version = 2

    allowed_range = (0.5, 0.72)
    variable = 's2_area_fraction_top'


class InteractionPeaksBiggest(ManyLichen):
    """Ensuring main peak is larger than the other peak
    
    (Should not be a big requirement for pax_v6.5.0)
    """
    version = 0

    def __init__(self):
        self.lichen_list = [self.S1(),
                            self.S2()]

    class S1(Lichen):
        def _process(self, df):
            df.loc[:, self.__class__.__name__] = df.s1 > df.largest_other_s1
            return df

    class S2(Lichen):
        def _process(self, df):
            df.loc[:, self.__class__.__name__] = df.s2 > df.largest_other_s2
            return df

class DAQVetoCut(ManyLichen):
    """Check if DAQ busy or HE veto
    
    Make sure no DAQ vetos happen during your event. This
    automatically checks both busy and high-energy vetos. 
    
    Requires Proximity minitrees.
    
    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:firstresults:daqandnoise
    
    Author: Daniel Coderre <daniel.coderre@lhep.unibe.ch>
    """
    version = 0

    def __init__(self):
        self.lichen_list = [self.BusyCheck(),
                            self.HEVCheck()]

    class BusyCheck(Lichen):
        def _process(self, df):
            df.loc[:, self.__class__.__name__] = abs(df['nearest_busy'])> df['event_duration']/2
            return df

    class HEVCheck(Lichen):
        def _process(self, df):
            df.loc[:, self.__class__.__name__] = abs(df['nearest_hev']) > df['event_duration']/2
            return df
                   
class SignalOverPreS2Junk(RangeLichen):
    """Cut events with lot of peak area before main S2
    
    (Currently not used)
    Compare S1 and S2 area to the area of other peaks before interaction S2

    This cut value is made up.... or at least found in a random notebook.
    """
    version = 0
    allowed_range = (0.5, 10)
    variable = 'signal_over_pre_s2_junk'

    def pre(self, df):
        df.loc[:, self.variable] = (df.s2 + df.s1) / (df.area_before_main_s2)
        return df


class S2SingleScatter(Lichen):
    """Check that largest other S2 is smaller than some bound...

    (Tianyu to add description and cut)
    """
    
    version = 0
    allowed_range = (0, np.inf)
    variable = 'temp'

    def other_s2_bound(self, s2):
        return np.clip((2 * s2) ** 0.5, 70, float('inf'))

    def _process(self, df):
        df.loc[:, self.__class__.__name__] = df.largest_other_s2 < self.other_s2_bound(df.s2)
        return df

    
class S1PatternLikelihood(Lichen):
    """Reject accidendal coicident events from lone s1 and lone s2.

       Details of the likelihood can be seen in the following note. Here, 97 quantile acceptance line estimated with Rn220 data (pax_v6.4.2) is used.
       https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:summary_note:s1_pattern_likelihood_cut
     
       Requires Extended minitrees.

       Author: Shingo Kazama <kazama@physik.uzh.ch>
    """

    version = 0

    def pre(self, df):
        df.loc[:,'temp'] = -2.39535 + 25.5857*pow(df['s1'], 0.5) + 1.30652*df['s1'] - 0.0638579*pow(df['s1'], 1.5)

    def _process(self, df):
        df.loc[:, self.__class__.__name__] = df['s1_pattern_fit'] < df.temp
        return df


class S2Width(ManyLichen):
    """S2 Width cut based on diffusion model.

    The S2 width cut compares the S2 width to what we could expect based on its
    depth in the detector. The inputs to this are the drift velocity and the
    diffusion constant. The allowed variation in S2 width is greater at low energy
    (since it is fluctuating statistically).

    The cut is roughly based on the observed distribution in AmBe and Rn data (paxv6.2.0)
    It is not described in any note, but you can see what it is doing in Sander's note here:
       https://xecluster.lngs.infn.it/dokuwiki/lib/exe/fetch.php?media=
       xenon:xenon1t:analysis:subgroup:backgrounds:meetings:170112_pb214_concentration_spectrum.html
    
    It should be applicable to data regardless of if it ER or NR; 
    above cS2 = 1e5 pe ERs the acceptance will go down due to track length effects.

    Author: Jelle, translation to lax by Chris.
    """
    version = 0

    def __init__(self):
        self.lichen_list = [self.S2WidthHigh(),
                            self.S2WidthLow()]

    def s2_width_model(self, z):
        diffusion_constant = PAX_CONFIG['WaveformSimulator']['diffusion_constant_liquid']
        v_drift = PAX_CONFIG['DEFAULT']['drift_velocity_liquid']

        w0 = 304 * units.ns
        return np.sqrt(w0 ** 2 - 3.6395 * diffusion_constant * z / v_drift ** 3)

    def subpre(self, df):
        # relative_s2_width
        df.loc[:, 'temp'] = df['s2_range_50p_area'] / S2Width.s2_width_model(self, df['z'])
        return df

    def relative_s2_width_bounds(s2, kind='high'):
        x = 0.3 * np.log10(np.clip(s2, 150, 7000))
        if kind == 'high':
            return 2.3 - x
        elif kind == 'low':
            return -0.3 + x
        raise ValueError("kind must be high or low")

    class S2WidthHigh(Lichen):
        def pre(self, df):
            return S2Width.subpre(self, df)

        def _process(self, df):
            df.loc[:, self.__class__.__name__] = (df.temp <= S2Width.relative_s2_width_bounds(df.s2,
                                                                                       kind='high'))
            return df

    class S2WidthLow(RangeLichen):
        def pre(self, df):
            return S2Width.subpre(self, df)

        def _process(self, df):
            df.loc[:, self.__class__.__name__] = (S2Width.relative_s2_width_bounds(df.s2,
                                                                            kind='low') <= df.temp)
            return df
