# -*- coding: utf-8 -*-
import numpy as np
from pax import units, configuration

PAX_CONFIG = configuration.load_configuration('XENON1T')
from lax.lichen import Lichen, RangeLichen, ManyLichen
from lax import __version__ as lax_version


class AllEnergy(ManyLichen):
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
        ]


class LowEnergy(AllEnergy):
    def __init__(self):
        AllEnergy.__init__(self)
        # Replaces Interaction exists
        self.lichen_list[1] = S1LowEnergyRange()

        # Use a simpler single scatter cut
        self.lichen_list[5] = S2SingleScatterSimple()

        self.lichen_list += [
            S1PatternLikelihood(),
            S2Width(),
            S1MaxPMT(),
        ]


class DAQVeto(ManyLichen):
    """Check if DAQ busy or HE veto

    Make sure no DAQ vetos happen during your event. This
    automatically checks both busy and high-energy vetos.

    Requires Proximity minitrees.

    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:firstresults:daqandnoise

    Contact: Daniel Coderre <daniel.coderre@lhep.unibe.ch>
    """
    version = 0

    def __init__(self):
        self.lichen_list = [self.BusyCheck(),
                            self.HEVCheck()]

    class BusyCheck(Lichen):
        def _process(self, df):
            df.loc[:, self.name()] = abs(df['nearest_busy']) > df['event_duration'] / 2
            return df

    class HEVCheck(Lichen):
        def _process(self, df):
            df.loc[:, self.name()] = abs(df['nearest_hev']) > df['event_duration'] / 2
            return df


class FiducialCylinder1T(ManyLichen):
    """Fiducial volume cut.

    The fidicual volume cut defines the region in depth and radius that we
    trust and use for the exposure. This is the region where the background
    distribution is flat.

    This version of the cut is based pax v6.2 bg run 0 data. See the
    note first results fiducial volume note for the study of the definition.

    Contact: Sander breur sanderb@nikhef.nl

    """
    version = 2

    def __init__(self):
        self.lichen_list = [self.Z(),
                            self.R()]

    class Z(RangeLichen):
        allowed_range = (-83.45, -13.45)
        variable = 'z'

    class R(RangeLichen):
        variable = 'r'  # Should add to minitrees

        def pre(self, df):
            df.loc[:, self.variable] = np.sqrt(df['x'] * df['x'] + df['y'] * df['y'])
            return df

        allowed_range = (0, 39.85)


class InteractionExists(RangeLichen):
    """Checks that there was a pairing of S1 and S2.

    Contact: Christopher Tunnell <tunnell@uchicago.edu>
    """
    version = 0
    allowed_range = (0, np.inf)
    variable = 'cs1'


class InteractionPeaksBiggest(ManyLichen):
    """Ensuring main peak is larger than the other peak

    (Should not be a big requirement for pax_v6.5.0)

    Contact: Christopher Tunnell <tunnell@uchicago.edu>
    """
    version = 0

    def __init__(self):
        self.lichen_list = [self.S1(),
                            self.S2()]

    class S1(Lichen):
        def _process(self, df):
            df.loc[:, self.name()] = df.s1 > df.largest_other_s1
            return df

    class S2(Lichen):
        def _process(self, df):
            df.loc[:, self.name()] = df.s2 > df.largest_other_s2
            return df


class S1LowEnergyRange(RangeLichen):
    """For isolating the low-energy band.

    Just an energy selection.

    Contact: Christopher Tunnell <tunnell@uchicago.edu>
    """
    version = 0
    allowed_range = (0, 200)
    variable = 'cs1'


class S1MaxPMT(Lichen):
    """Cut events which have a high fraction of the area in a single PMT

    Cuts events which are mostly seen by one PMT. These events could be for
    example afterpulses or light emission. This is the 99% quantile fit using
    pax 6.4.2 on Rn220.

    xenon:xenon1t:yuehuan:analysis:0sciencerun_s1_pmtmax

    Contact: Julien Wulf <jwulf@physik.uzh.ch>
    """
    version = 0

    def pre(self, df):
        df.loc[:, 'temp'] = 0.052 * df['s1'] + 4.15

    def _process(self, df):
        df.loc[:, self.name()] = df['s1_largest_hit_area'] < df.temp
        return df


class S1PatternLikelihood(Lichen):
    """Reject accidendal coicident events from lone s1 and lone s2.

    Details of the likelihood can be seen in the following note. Here, 97
    quantile acceptance line estimated with Rn220 data (pax_v6.4.2) is used.

       xenon:xenon1t:analysis:summary_note:s1_pattern_likelihood_cut

    Requires Extended minitrees.

    Contact: Shingo Kazama <kazama@physik.uzh.ch>
    """

    version = 0

    def pre(self, df):
        df.loc[:, 'temp'] = -2.39535 + 25.5857 * pow(df['s1'], 0.5) + 1.30652 * df['s1'] - 0.0638579 * pow(df['s1'],
                                                                                                           1.5)

    def _process(self, df):
        df.loc[:, self.name()] = df['s1_pattern_fit'] < df.temp
        return df


class S1SingleScatter(Lichen):
    """Requires only one valid interaction between the largest S2, and any S1 recorded before it.

    The S1 cut checks that any possible secondary S1s recorded in a waveform, could not have also
    produced a valid interaction with the primary S2. To check whether an interaction between the
    second largest S1 and the largest S2 is valid, we use the S2Width cut. If the event would pass
    the S2Width cut, a valid second interaction exists, and we may have mis-identified which S1 to
    pair with the primary S2. Therefore we cut this event. If it fails the S2Width cut the event is
    not removed.

    Current version is developed on unblinded Bkg data (paxv6.4.2). It is described in this note:
	https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:jacques:s1_single_scatter_cut

    It should be applicable to data regardless of if it is ER or NR.

    Contact: Jacques <jpienaa@purdue.edu>
    """

    version = 0

    def _process(self, df):
        s2width = S2Width

        alt_rel_width = df['s2_range_50p_area'] / s2width.s2_width_model(df['alt_s2_interaction_z'])
        alt_interaction_passes = alt_rel_width < s2width.relative_s2_width_bounds(df.s2.values, kind='high')
        alt_interaction_passes &= alt_rel_width > s2width.relative_s2_width_bounds(df.s2.values, kind='low')

        df.loc[:, (self.name())] = True ^ alt_interaction_passes
        return df


class S2AreaFractionTop(Lichen):
    """Cuts events with an unusual fraction of S2 on top array.

    Primarily cuts gas events with a particularly large S2 AFT, also targets some
    strange / junk / other events with a low AFT.

    This cut has been checked on S2 ranges between 0 and 50 000 pe.

    Described in the note at: xenon:xenon1t:analysis:firstresults:s2_aft_cut_summary

    Contact: Adam Brown <abrown@physik.uzh.ch>
    """

    def _process_v2(self, df):
        """This is a simple range cut which was chosen by eye.
        """
        allowed_range = (0.5, 0.72)
        aft_variable = 's2_area_fraction_top'
        df.loc[:, self.name()] = ((df[aft_variable] < allowed_range[1]) &
                                  (df[aft_variable] > allowed_range[0]))
        return df

    def _process_v3(self, df):
        """This is a more complex and much tighter cut than version 2 from fitting
        the distribution in slices in S2 space and choosing the 0.5% and 99.5% quantile
        for each fit to give a theoretical acceptance of 99%.
        """

        def upper_limit_s2_aft(s2):
            return 0.6177399420527526 + 3.713166211522462e-08 * s2 + 0.5460484265254656 / np.log(s2)

        def lower_limit_s2_aft(s2):
            return 0.6648160611018054 - 2.590402853814859e-07 * s2 - 0.8531029789184852 / np.log(s2)

        aft_variable = 's2_area_fraction_top'
        s2_variable = 's2'
        df.loc[:, self.name()] = ((df[aft_variable]
                                   < upper_limit_s2_aft(df[s2_variable])) &
                                  (df[aft_variable]
                                   > lower_limit_s2_aft(df[s2_variable])))

        return df

    def __init__(self, version=2):
        self.version = version
        if not version in [2, 3]:
            raise ValueError('Only versions 2 and 3 are implemented')

    def _process(self, df):
        if self.version == 2:
            return self._process_v2(df)
        elif self.version == 3:
            return self._process_v3(df)
        else:
            raise ValueError('Only versions 2 and 3 are implemented')


class S2SingleScatter(Lichen):
    """Check that largest other S2 area is smaller than some bound.

    The single scatter is to cut an event if its largest_other_s2 is too large.
    As the largest_other_s2 takes a greater value when they originated from some real scatters
    in comparison, those from photo-ionization in single scatter cases would be smaller.

    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:firstresults:cut:s2single

    Contact: Tianyu Zhu <tz2263@columbia.edu>
    """

    version = 2
    allowed_range = (0, np.inf)
    variable = 'temp'

    def other_s2_bound(self, s2):
        y0 = s2 * 0.00832 + 72.3
        y1 = s2 * 0.03 - 109

        d0 = 1 / (np.exp((s2 - 23300) * 5.91e-4) + 1)
        d1 = 1 / (np.exp((23300 - s2) * 5.91e-4) + 1)

        return y0 * d0 + y1 * d1

    def _process(self, df):
        df.loc[:, self.name()] = df.largest_other_s2 < self.other_s2_bound(df.s2)
        return df


class S2SingleScatterSimple(Lichen):
    """Check that largest other S2 area is smaller than some bound.

    It's the low energy limit of the S2SingleScatter Cut
    applies to S2 < 20000

    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:firstresults:cut:s2single

    Contact: Tianyu Zhu <tz2263@columbia.edu>
    """
    version = 0
    allowed_range = (0, np.inf)
    variable = 'temp'

    def other_s2_bound(self, s2):
        return s2 * 0.00832 + 72.3

    def _process(self, df):
        df.loc[:, self.name()] = df.largest_other_s2 < self.other_s2_bound(df.s2)
        return df


class S2Threshold(RangeLichen):
    """The S2 energy at which the trigger is perfectly efficient.

    See: https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:firstresults:daqtriggerpaxefficiency

    Contact: Jelle Aalbers <aalbers@nikhef.nl>
    """
    version = 0
    allowed_range = (150, np.inf)
    variable = 's2'


class S2Width(ManyLichen):
    """S2 Width cut based on diffusion model

    The S2 width cut compares the S2 width to what we could expect based on its depth in the detector. The inputs to
    this are the drift velocity and the diffusion constant. The allowed variation in S2 width is greater at low
    energy (since it is fluctuating statistically) Ref: (arXiv:1102.2865)

    It should be applicable to data regardless of if it ER or NR;
    above cS2 = 1e5 pe ERs the acceptance will go down due to track length effects.

    Tune the diffusion model parameters based on pax v6.4.2 AmBe data according to note:

    xenon:xenon1t:yuehuan:analysis:0sciencerun_s2width_update0#comparison_with_diffusion_model_cut_by_jelle_pax_v642

    Contact: Yuehuan <weiyh@physik.uzh.ch>
    """
    version = 1

    def __init__(self):
        self.lichen_list = [self.S2WidthHigh(),
                            self.S2WidthLow()]

    @staticmethod
    def s2_width_model(z):
        diffusion_constant = PAX_CONFIG['WaveformSimulator']['diffusion_constant_liquid']
        v_drift = PAX_CONFIG['DEFAULT']['drift_velocity_liquid']

        w0 = 348.6 * units.ns
        return np.sqrt(w0 ** 2 - 4.0325 * diffusion_constant * z / v_drift ** 3)

    def subpre(self, df):
        # relative_s2_width
        df.loc[:, 'temp'] = df['s2_range_50p_area'] / S2Width.s2_width_model(df['z'])
        return df

    @staticmethod
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
            df.loc[:, self.name()] = (df.temp <= S2Width.relative_s2_width_bounds(df.s2,
                                                                                  kind='high'))
            return df

    class S2WidthLow(RangeLichen):
        def pre(self, df):
            return S2Width.subpre(self, df)

        def _process(self, df):
            df.loc[:, self.name()] = (S2Width.relative_s2_width_bounds(df.s2,
                                                                       kind='low') <= df.temp)
            return df


class SignalOverPreS2Junk(RangeLichen):
    """Cut events with lot of peak area before main S2 (currently working for small s2s)

    Not used.  Compare S1 and S2 area to the area of other peaks before interaction S2

    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:yuehuan:analysis:0sciencerun_signal_noise

    Contact: Julien Wulf <jwulf@physik.uzh.ch>
    """
    version = 0
    allowed_range = (0, 1)
    variable = 'signal_over_pre_s2_junk'

    def pre(self, df):
        df.loc[:, self.variable] = (df.area_before_main_s2 - df.s1) / (df.s2 + df.s1)
        return df
