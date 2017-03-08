=======
History
=======

0.8.1 (2017-03-07)
------------------

* Update cut list

lax 0.8.1

CutAllEnergy
  * CutFiducialCylinder1T v3
  * CutInteractionExists v0
  * CutS2Threshold v0
  * CutInteractionPeaksBiggest v0
  * CutS2AreaFractionTop v2
  * CutS2SingleScatter v2
  * CutDAQVeto v0
  * CutS1SingleScatter v1
  * CutS1AreaFractionTop v1
  * CutS2PatternLikelihood v0
CutLowEnergy
  * CutFiducialCylinder1T v3
  * CutS1LowEnergyRange v0
  * CutS2Threshold v0
  * CutInteractionPeaksBiggest v0
  * CutS2AreaFractionTop v2
  * CutS2SingleScatterSimple v0
  * CutDAQVeto v0
  * CutS1SingleScatter v1
  * CutS1AreaFractionTop v1
  * CutS2PatternLikelihood v0
  * CutS1PatternLikelihood v0
  * CutS2Width v2
  * CutS1MaxPMT v0
  * CutSignalOverPreS2Junk v1
  * CutSingleElectronS2s v0

0.8.0 (2017-03-07)
------------------

* Update area before main S2 < 300

0.7.0 (2017-03-07)
------------------

* Update 1T FV (#40)
* Tune S2 width cut (#38)
* S1 width cut (#33)
* S2 pattern likelihood (#34)
* S1 AFT speed fix (#32)

0.6.2 (2017-03-03)
------------------

* Added DistanceToAmBe cut (#31)

0.6.1 (2017-03-02)
------------------

* Fix bug in how data file for S1 AFT loaded.

0.6.0 (2017-03-02)
------------------

* Fix problem in S1 single scatter definition (#26)
* S1 Area fraction top included (#16)

CutAllEnergy
  * CutFiducialCylinder1T v2
  * CutInteractionExists v0
  * CutS2Threshold v0
  * CutInteractionPeaksBiggest v0
  * CutS2AreaFractionTop v2
  * CutS2SingleScatter v2
  * CutDAQVeto v0
  * CutS1SingleScatter v1
  * CutS1AreaFractionTop v0
CutLowEnergy
  * CutFiducialCylinder1T v2
  * CutS1LowEnergyRange v0
  * CutS2Threshold v0
  * CutInteractionPeaksBiggest v0
  * CutS2AreaFractionTop v2
  * CutS2SingleScatterSimple v0
  * CutDAQVeto v0
  * CutS1SingleScatter v1
  * CutS1AreaFractionTop v0
  * CutS1PatternLikelihood v0
  * CutS2Width v1
  * CutS1MaxPMT v0

0.5.3 (2017-02-28)
------------------

* Another pre() error

0.5.2 (2017-02-28)
------------------

* S1 Pattern and max PMT had error in pre() not returning df
* ManyLichen print list of cuts works

0.5.1 (2017-02-28)
------------------

* Fix SignalOverPreS2Junk key (#24)

0.5.0 (2017-02-28)
------------------

* Doc improvements.
* S1 Pattern likelihood in LowEnergyCuts (#21)
* Max PMT in S1 (v0) LowEnergyCuts (#15)
* S2AreaFractionTopCut now can have v3 (v2 still default) with tighter AFT selection (#14)
* SignalOverPreS2Junk v0, not used (#20)
* S2SingleScatter in all cuts, S2SingleScatterSimple in LowEnergy (#9)
* Tune S2 width (#18)
* S1 Single Scatter (#22)

List of current cuts:

CutAllEnergy
	CutFiducialCylinder1T version 2
	CutInteractionExists version 0
	CutS2Threshold version 0
	CutInteractionPeaksBiggest version 0
	CutS2AreaFractionTop version 2
	CutS2SingleScatter version 2
	CutDAQVeto version 0
	CutS1SingleScatter version 0
CutLowEnergy
	CutFiducialCylinder1T version 2
	CutS1LowEnergyRange version 0
	CutS2Threshold version 0
	CutInteractionPeaksBiggest version 0
	CutS2AreaFractionTop version 2
	CutS2SingleScatterSimple version 0
	CutDAQVeto version 0
	CutS1SingleScatter version 0
	CutS1PatternLikelihood version 0
	CutS2Width version 1
	CutS1MaxPMT version 0


0.4.0 (2017-02-24)
------------------

* Add DAQ busy and HE veto requirement that requires Proximity tree (#7)

0.3.0 (2017-02-21)
------------------

* Update s2_area_fraction_top cut (#5)
* Improve docs (#4)
* Plotting arbitrary axes

0.2.2 (2017-02-21)
------------------

* Tweaks

0.2.1 (2017-02-21)
------------------

* Remove signal noise cut since doesn't work

0.2.0 (2017-02-21)
------------------

* Bug where all cuts not applied properly

0.1.6 (2017-02-20)
------------------

* Add signal noise

0.1.5 (2017-02-20)
------------------

* Fix fiducial volume

0.1.4 (2017-02-20)
------------------

* Reorder cuts again

0.1.3 (2017-02-20)
------------------

* Update requirements

0.1.2 (2017-02-20)
------------------

* Reorder cuts and save some intermediates ('r')

0.1.1 (2017-02-20)
------------------

* Cut versioning

0.1.0 (2017-02-19)
------------------

* First release on PyPI.
* Initial cuts for SR0.
