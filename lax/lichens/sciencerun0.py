"""The cuts for science run 0

This includes all current definitions of the cuts for the first science run
"""

# -*- coding: utf-8 -*-
import inspect
import os
import json

import numpy as np
from pax import units, configuration

from scipy.interpolate import RectBivariateSpline
from scipy.stats import binom_test
from scipy import interpolate
import json

PAX_CONFIG = configuration.load_configuration('XENON1T')
from lax.lichen import Lichen, RangeLichen, ManyLichen, StringLichen
from lax import __version__ as lax_version

# Store the directory of our data files
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
                        '..', 'data')


class AllEnergy(ManyLichen):
    """Cuts applicable for low and high energy (gammas)

    This is a subset mostly of the low energy cuts.
    """
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
            S1AreaFractionTop(),
            S2PatternLikelihood(),
            S2Tails()
        ]


class LowEnergy(AllEnergy):
    """Select events with cs1<200

    This is the list that we'll use for the actual DM search, therefore
    those energies.
    """

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
#            SignalOverPreS2Junk(),
            SingleElectronS2s()
        ]


class DAQVeto(ManyLichen):
    """Check if DAQ busy or HE veto

    Make sure no DAQ vetos happen during your event. This
    automatically checks both busy and high-energy vetos.
    Also makes sure last BUSY type is 'off' and cuts
    last 21 seconds of each run.

    Requires Proximity minitrees.

    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:firstresults:daqandnoise

    Contact: Daniel Coderre <daniel.coderre@lhep.unibe.ch>
    """
    version = 1

    def __init__(self):
        self.lichen_list = [self.EndOfRunCheck(),
                            self.BusyTypeCheck(),
                            self.BusyCheck(),
                            self.HEVCheck()]

    class EndOfRunCheck(Lichen):
        """Check that the event does not come in the last 21 seconds of the run
        """
        def _process(self, df):
            df_runs = df.loc[df.reset_index().groupby(['run_number'])['event_time'].idxmax()]
            runvals = {}
            for _, row in df_runs.iterrows():
                runvals[row['run_number']] = row['event_time']
            df.loc[:, self.name()] = df.apply(lambda row: row['event_time'] <
                                              runvals[row['run_number']] - 21e9, axis=1)
            return df

    class BusyTypeCheck(Lichen):
        """Ensure that the last busy type (if any) is OFF
        """
        def _process(self, df):
            df.loc[:, self.name()] = ((~(df['previous_busy_on'] < 60e9)) |
                                  (df['previous_busy_off'] <
                                   df['previous_busy_on']))
            return df

    class BusyCheck(Lichen):
        """Check if the event contains a BUSY veto trigger
        """
        def _process(self, df):
            df.loc[:, self.name()] = (abs(df['nearest_busy']) >
                                      df['event_duration'] / 2)
            return df

    class HEVCheck(Lichen):
        """Check if the event contains a HE veto trigger
        """
        def _process(self, df):
            df.loc[:, self.name()] = (abs(df['nearest_hev']) >
                                      df['event_duration'] / 2)
            return df


class S2Tails(Lichen):
    """Check if event is in a tail of a previous S2

    Requires S2Tail minitrees.

    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:subgroup:wimphysics:s2_tails_sr0

    Contact: Daniel Coderre <daniel.coderre@lhep.unibe.ch>
    """
    
    def _process(self, df):
        df.loc[:, self.name()] = ((~(df['s2_over_tdiff'] >= 0)) |
                                    (df['s2_over_tdiff'] < 0.04))
        return df

    
class FiducialCylinder1T(StringLichen):
    """Fiducial volume cut.

    The fidicual volume cut defines the region in depth and radius that we
    trust and use for the exposure. This is the region where the background
    distribution is flat.

    This version of the cut is based pax v6.4 bg run 0 data. See the
    note first results fiducial volume note for the study of the definition.

    Contact: Sander breur <sanderb@nikhef.nl>

    """
    version = 3
    string = "(-92.9 < z) & (z < -9) & (r < 36.94)"

    def pre(self, df):
        df.loc[:, 'r'] = np.sqrt(df['x'] * df['x'] + df['y'] * df['y'])
        return df

class FiducialFourLeafClover1250kg(StringLichen):
    """Fiducial volume cut: Four leaf Clover
    
    Our FV is constraint by two depth planes (4 cm above the gate, 9 cm below the gate)
    and a curved surface which stays 4 cm from the measured walls of the tpc.

    The fidicual volume cut defines the region in depth and radius that we
    trust and could use for the first science run. This is the region where
    we still expect the background distribution is flat.

    For the phi dependent shape we took the 210Po data from wall events and
    calculated where the ... percentile of events leaking in. For the angle
    the edge of the event distribution for different z slices where taken.
    The angel was taken so that both the lower and upper part of the FV stay a similar
    distance away from the edge.

    This version of the cut is based pax v6.4 bg run 0 data. See the
    note first results fiducial volume note for the study of the definition.

    Contact: Sander breur <sanderb@nikhef.nl>

    """
    version = 1

    string = "(-92.9 < z) & (z < -9) & (r_phi < r_max)"

    def pre(self, df):

        # first get the points from 210Po
        # open file and get text
        rho_phi_filename = os.path.join(DATA_DIR,
                                       'R_phi_curve_360points.txt')
        with open(rho_phi_filename) as file:
            text = file.readlines()
        # get value per line
        phi_values = []
        r_values = []
        for line in text:
            phi_values.append(float(line.split()[0]))
            r_values.append(float(line.split()[1]))

        # Set values needed for phi dependent radius
        phi_values = np.array(phi_values)
        r_values = np.array(r_values)
        # this is the average radius for the shape, this we scale
        average_radius_egg = np.average(r_values)

        # Set user defined values for the FV:
        # Set the radius and angle
        radius_scaling_value = 41  # cm # this sets the average radius
        radius_offset_value = 1  # cm # this sets the angle
        # Set the height
        depth_upper_bound = -9  # cm - exlude all gamma's
        depth_lower_bound = -96.9 + 4  # cm - stay away from the cathode
        max_height = depth_upper_bound - depth_lower_bound

        # Go to polar coordinates
        def cart2pol(x_value, y_value):
            """This is a simple function to change coordinate system.
            """
            rho = np.sqrt(x_value ** 2 + y_value ** 2)
            phi = np.arctan2(y_value, x_value)
            return (rho, phi)

        # Find argument of nearest value in array
        def find_nearest(array, values):
            """This is a simple function to find the argument of a value in an array.
            """
            indices = np.abs(np.subtract.outer(array, values)).argmin(0)
            return indices

        # Get the dep max radius for a FV in R with an angle set by r_offset
        # takes depth array [cm], Max radius [cm], radius offset [cm],
        # total height of cylinder [cm], center of cylinder in depth [cm]
        def coffee_r(z_value, R, r_offset, height, z_center):
            """This make the radius depth dependent. For a given r_offset it increases
            the top radius by r_offset and decreases the bottom radius by r_offset while
            keeping a straight line in the R2-Z space to keep the volume the same.
            """
            return np.sqrt((((R + r_offset) ** 2 - (R - r_offset) ** 2) / height) * (z_value - z_center + (height / 2)) + (
                R - r_offset) ** 2)  # returns radius array [cm]

        # Rho from data
        df.loc[:, 'r_phi'] = cart2pol(df['x'], df['y'])[0]
        # Max Rho
        df.loc[:, 'r_max'] = ((radius_scaling_value/ average_radius_egg) *
                                coffee_r(df['z'],
                                r_values[find_nearest(phi_values, cart2pol(df['x'], df['y'])[1])],
                                radius_offset_value,
                                max_height,
                                -max_height / 2 + depth_upper_bound))
        return df


class DistanceToAmBe(StringLichen):
    """AmBe Fiducial volume cut.
    This uses the same Z cuts as the 1T fiducial cylinder, but a wider allowed range in R to maximize the number of nuclear recoils.
    There is a third cut on the distance to the source, so that we cut away background ER.
    Link to note:
    https://xecluster.lngs.infn.it/dokuwiki/lib/exe/fetch.php?media=xenon:xenon1t:hogenbirk:nr_band_sr0.html

    Contact: Erik Hogenbirk <ehogenbi@nikhef.nl>

    """
    version = 1
    string = "(distance_to_source < 80)"

    def pre(self, df):
        source_position = (55.965311731903, 43.724893639103577, -50)
        df.loc[:, 'distance_to_source'] = ((source_position[0] - df['x']) ** 2 +
                                           (source_position[1] - df['y']) ** 2 +
                                           (source_position[2] - df['z']) ** 2) ** 0.5
        return df


class InteractionExists(StringLichen):
    """Checks that there was a pairing of S1 and S2.

    Contact: Christopher Tunnell <tunnell@uchicago.edu>
    """
    version = 0
    string = "0 < cs1"


class InteractionPeaksBiggest(StringLichen):
    """Ensuring main peak is larger than the other peak

    (Should not be a big requirement for pax_v6.5.0)

    Contact: Christopher Tunnell <tunnell@uchicago.edu>
    """
    version = 0
    string = "(s1 > largest_other_s1) & (s2 > largest_other_s2)"


class S1LowEnergyRange(RangeLichen):
    """For isolating the low-energy band.
    Just an energy selection.
    Contact: Christopher Tunnell <tunnell@uchicago.edu>
    """
    version = 0
    allowed_range = (0, 200)
    variable = 'cs1'


class S1MaxPMT(StringLichen):
    """Cut events which have a high fraction of the area in a single PMT

    Cuts events which are mostly seen by one PMT. These events could be for
    example afterpulses or light emission. This is the 99% quantile fit using
    pax 6.4.2 on Rn220.

    xenon:xenon1t:yuehuan:analysis:0sciencerun_s1_pmtmax

    Contact: Julien Wulf <jwulf@physik.uzh.ch>
    """
    version = 0
    string = "s1_largest_hit_area < 0.052 * s1 + 4.15"


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
        df.loc[:, 'temp'] = -2.39535 + \
                            25.5857 * pow(df['s1'], 0.5) + \
                            1.30652 * df['s1'] - \
                            0.0638579 * np.power(df['s1'], 1.5)
        return df

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

    version = 1

    def _process(self, df):
        s2width = S2Width

        alt_rel_width = df['s2_range_50p_area'] / s2width.s2_width_model(df['alt_s1_interaction_z'])
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
        if version not in [2, 3]:
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

    @classmethod
    def other_s2_bound(cls, s2_area):
        rescaled_s2_0 = s2_area * 0.00832 + 72.3
        rescaled_s2_1 = s2_area * 0.03 - 109

        another_term_0 = 1 / (np.exp((s2_area - 23300) * 5.91e-4) + 1)
        another_term_1 = 1 / (np.exp((23300 - s2_area) * 5.91e-4) + 1)

        return rescaled_s2_0 * another_term_0 + rescaled_s2_1 * another_term_1

    def _process(self, df):
        df.loc[:, self.name()] = df.largest_other_s2 < self.other_s2_bound(df.s2)
        return df


class S2SingleScatterSimple(StringLichen):
    """Check that largest other S2 area is smaller than some bound.

    It's the low energy limit of the S2SingleScatter Cut
    applies to S2 < 20000

    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:firstresults:cut:s2single

    Contact: Tianyu Zhu <tz2263@columbia.edu>
    """
    version = 0
    string = 'largest_other_s2 < s2 * 0.00832 + 72.3'


class S2PatternLikelihood(StringLichen):
    """Reject poorly reconstructed S2s and multiple scatters.

    Details of the likelihood can be seen in the following note. Here, 98
    quantile acceptance line estimated with Rn220 data (pax_v6.4.2) is used.

       xenon:xenon1t:analysis:firstresults:s2_pattern_likelihood_cut

    Requires Extended minitrees.

    Contact: Bart Pelssers  <bart.pelssers@fysik.su.se>
    """
    version = 0
    string = "s2_pattern_fit < 75 + 10 * s2**0.45"


class S2Threshold(StringLichen):
    """The S2 energy at which the trigger is perfectly efficient.

    See: https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:firstresults:daqtriggerpaxefficiency

    Contact: Jelle Aalbers <aalbers@nikhef.nl>
    """
    version = 0
    string = "150 < s2"


class S2Width(ManyLichen):
    """S2 Width cut based on diffusion model

    The S2 width cut compares the S2 width to what we could expect based on its depth in the detector. The inputs to
    this are the drift velocity and the diffusion constant. The allowed variation in S2 width is greater at low
    energy (since it is fluctuating statistically) Ref: (arXiv:1102.2865)

    It should be applicable to data regardless of if it ER or NR;
    above cS2 = 1e5 pe ERs the acceptance will go down due to track length effects.

    Tune the diffusion model parameters based on pax v6.4.2 AmBe data according to note:

    xenon:xenon1t:yuehuan:analysis:0sciencerun_s2width_update0#comparison_with_diffusion_model_cut_by_jelle_pax_v642

    Contact: Yuehuan <weiyh@physik.uzh.ch>, Jelle <jaalbers@nikhef.nl>
    """
    version = 2

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
        x = 0.5 * np.log10(np.clip(s2, 150, 4500 if kind == 'high' else 2500))
        if kind == 'high':
            return 3 - x
        elif kind == 'low':
            return -0.9 + x
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


class S1AreaFractionTop(RangeLichen):
    '''S1 area fraction top cut

    Uses scipy.stats.binom_test to compute a p-value based on the
    observed number of s1 photons in the top array, given the expected
    probability that a photon at the event's (r,z) makes it to the top array.

    Uses a 3D map generated with Kr83m 32 keV line

    note: https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:darryl:xe1t_s1_aft_map

    Author: Darryl Masson, dmasson@purdue.edu
    '''

    version = 1
    variable = 'pvalue_s1_area_fraction_top'
    allowed_range = (1e-4, 1 + 1e-7)  # must accept p-value = 1.0 with a < comparison

    def __init__(self):
        aftmap_filename = os.path.join(DATA_DIR,
                                       's1_aft_rz_02Mar2017.json')
        with open(aftmap_filename) as data_file:
            data = json.load(data_file)
        r_pts = np.array(data['r_pts'])
        z_pts = np.array(data['z_pts'])
        aft_vals = np.array([data['map'][i*len(z_pts):(i+1)*len(z_pts)] for i in range(len(r_pts))]) # unpack 1d array to 2d
        self.aft_map = RectBivariateSpline(r_pts,z_pts,aft_vals)

    def pre(self, df):
        df.loc[:, self.variable] = df.apply(lambda row: binom_test(np.round(row['s1_area_fraction_top'] * row['s1']),
                                                                   np.round(row['s1']),
                                                                   self.aft_map(np.sqrt(row['x']**2 + row['y']**2),
                                                                                row['z'])[0,0]),
                                            axis=1)
        return df


class SignalOverPreS2Junk(StringLichen):
    """Cut events with lot of peak area before main S2 

    Contact: Julien Wulf <jwulf@physik.uzh.ch>
    """
    version = 1
    string = "area_before_main_s2 - s1 < 300"

    
class SingleElectronS2s(Lichen):
    """Remove mis-identified single electron S2s classified as S1s

    Details of the definition can be seen in the following note:

    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:firstresults:exploring_se_cut
    
    This was done by redrawing and improving the classification bounds for S1s at low energies by
    building up from Jelles low-energy classification work at a peaks level.
    To do this Rn220 data processed in pax.v6.4.2 was used.
    
    Requires: TotalProperties, LowEnergyS1Candidates minitrees.
    
    Contact: Miguel Angel Vargas <m_varg03@uni-muenster.de>
    """
    version = 0
    allowed_range_area = (10, 200)
    allowed_range_rt =(11,450)
    area_variable = 's1'
    rt_variable = 's1_rise_time'
    aft_variable = 's1_area_fraction_top'

    bound_v4 = interpolate.interp1d([0, 0.3, 0.4, 0.5, 0.60, 0.60],[70, 70, 61, 61,35,0],
                                    fill_value='extrapolate', kind='linear')

    def _process(self, df):
        # Is the event inside the area box considered for this study?
        cond = ((df[self.area_variable] < self.allowed_range_area[0]) &
                (df[self.area_variable] > self.allowed_range_area[1]) &
                (df[self.rt_variable] > self.allowed_range_rt[0]) &
                (df[self.rt_variable] < self.allowed_range_rt[1]))

        # Pass events by default
        passes = np.ones(len(df), dtype=np.bool)

        # Reject events inside the box that don't pass the bound
        passes[cond] = df[self.rt_variable][cond] < SingleElectronS2s.bound_v4(df[self.aft_variable][cond])

        df.loc[:, self.name()] = passes
        return df
