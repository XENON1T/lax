=======
History
=======

1.2.1 (2017-11-27)
------------------

* Muon Veto cut added (#55)
* Update on S1 Pattern Likelihood Cut (#99)
* Add super elliptical fiducial volumes for testing in SR1 (#100)

1.2.0 (2017-11-20)
------------------
Interim	release	for tracking data used in preliminary fits
(https://github.com/XENON1T/bbf/pull/35)
(https://github.com/XENON1T/SR1Results/pull/2)
 
* S1 area fraction top update and relegate to LowE (#56, #83)
* Update S2WidthCut and parameters for SR1 (#62, #75, #77)
* Fix LowEnergyRn220 accidentally removing DAQVeto cut (#66)
* Fix bug and treatment of 'nan' in S2SingleScatter (#69, #74)
* Reinstate InteractionPeaksBiggest (#71)
* Update S2Tail cut for SR1 (#76)
* S2PatternLikelihood Cut Tuning (#80, #87)
* Fix AmBe source position (#58, #97)
* Remove AmBe/NGFiducial cuts (#86, #98)
* Temporary 1.3T FV (#89, #94)
* Add Corrections minitree and SR1 switch to laxer (#64)
* Documentation (#73, #92)

1.0.0 (2017-06-22)
------------------

* Start SR1 lichens (#58)
* Neutron gun and updated AmBe FV (Also in #58)
* MC compatability (#57)
* Remove InteractionPeakBiggest for SR1 since done by pax

0.11.1 (2017-03-29)
------------------

* Refix SingleElectronS2 (#51)

0.11.0 (2017-03-29)
------------------

* Fix SingleElectronS2 (#50)

0.10.2 (2017-03-27)
------------------

* Bug in lichen definition (append not used as function)

0.10.0 (2017-03-24)
------------------

* Fix bug in SingleElectron lichen definition
* Create LowEnergyBackground, LowEnergyRn220, LowEnergyAmBe to simplfy cut application

0.9.5 (2017-03-23)
------------------

* sqrt(x*x + y*y) instead of r

0.9.4 (2017-03-10)
------------------

* Up version on S2 threshold

0.9.3 (2017-03-10)
------------------

* Raise S2 threshold to 200 PE

0.9.2 (2017-03-09)
------------------

* AmBe fiducial change to include z/r cut.
* Get runtime from hax for DAQ cut (#43)
* Fix warning in SingleElectronS2s (#44)

0.9.1 (2017-03-08)
------------------

* Fix to DAQ cut (#42)

0.9.0 (2017-03-08)
------------------

* DAQ cut update (#39)

0.8.6 (2017-03-08)
------------------

* Add larger FV option (#41)

0.8.5 (2017-03-07)
------------------

* Remove junk cut from main list since used for calibrations.

0.8.4 (2017-03-07)
------------------

* Fix bug in S1 width cut

0.8.3 (2017-03-07)
------------------

...

0.8.2 (2017-03-07)
------------------

* Fix bug in junk cut definition

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
