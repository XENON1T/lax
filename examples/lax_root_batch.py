"""Make ROOT file of all lax output

Usage with real data:
     python lax_root_batch.py --RUN_NUMBER 6731 -a-pax_version 6.6.5 \
         --minitree_path /project/lgrandi/xenon1t/minitrees/pax_v6.6.5

Usage with MC:
    python lax_root_batch.py --RUN_NUMBER -1 --pax_version 6.6.5 \
         --minitree_path output --filename Xenon1T_TPC_Rn222_00000_g4mc_G4_Sort_pax
"""
import argparse
import sys

import hax
from lax.lichens import sciencerun0

import root_pandas


parser = argparse.ArgumentParser(description="Create lichen ROOT files with lax")

parser.add_argument('--RUN_NUMBER', dest='RUN_NUMBER',
                    action='store', required=True, type=int,
                    help='Run number to process (-1 for MC)')

parser.add_argument('--pax_version', dest='PAX_VERSION',
                    action='store', required=True,
                    help='pax version to process')

parser.add_argument('--minitree_path', dest='MINITREE_PATH',
                    action='store', required=True,
                    help='Path to hax minitrees')

parser.add_argument('--filename', dest='FILENAME',
                    action='store', required=False,
                    help='Name of pax file (without .root)')

parser.add_argument('--OUTPUT_PATH', dest='OUTPUT_PATH',
                    action='store', required=False, default='',
                    help='Name of output file (without .root)')

args = parser.parse_args(sys.argv[1:])

PAX_VERSION_POLICY = args.PAX_VERSION
RUN_NUMBER = args.RUN_NUMBER
MINITREE_NAMES = ['Fundamentals', 'Basics', 'TotalProperties',
                  'Extended', 'TailCut', 'Proximity']
OUTPUT_PATH = args.OUTPUT_PATH

LAX_LICHENS = [sciencerun0.AllEnergy(),
               sciencerun0.LowEnergyRn220(),
               sciencerun0.LowEnergyAmBe(),
               sciencerun0.LowEnergyBackground()]

TREENAME = 'tree'

# MC
if args.RUN_NUMBER < 0:

    # No run dependent sims yet
    PAX_VERSION_POLICY = 'loose'

    # Use filename instead of run number
    RUN_NUMBER = args.MINITREE_PATH+'/'+args.FILENAME

    # Remove meaningless variables
    MINITREE_NAMES.remove('TailCut')
    MINITREE_NAMES.remove('Proximity')

    # Remove meaningless cuts (HARDCODE WARNING)
    for cuts in LAX_LICHENS:
        cuts.lichen_list.pop(6)  # DAQVeto
        cuts.lichen_list.pop(9)  # S2Tails

    TREENAME += 'mc'

    if OUTPUT_PATH is '':
        OUTPUT_PATH = args.FILENAME+"_lax"

if OUTPUT_PATH is '':
    OUTPUT_PATH = "%d_lax" % RUN_NUMBER

# Initialize hax
HAX_KWARGS = {'experiment': 'XENON1T',
              'pax_version_policy': PAX_VERSION_POLICY,
              'minitree_paths': ['.', args.MINITREE_PATH]
             }

# Be careful what you're doing here
if args.RUN_NUMBER < 0:
    HAX_KWARGS['blinding_cut'] = 'RUN_NUMBER<=0'

HAX_KWARGS['runs_url'] = 'mongodb://eb:{password}@xenon1t-daq.lngs.infn.it:27017/run'

hax.init(**HAX_KWARGS)

DF_ALL = hax.minitrees.load(RUN_NUMBER, MINITREE_NAMES)

for cuts in LAX_LICHENS:

    DF_ALL = cuts.process(DF_ALL)

DF_ALL.to_root(OUTPUT_PATH+'.root', TREENAME)
