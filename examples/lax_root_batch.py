"""Make ROOT file of all lax output

Usage with real data:
     python lax_root_batch.py --run_number 6731 --pax_version 6.6.5 --minitree_path /project/lgrandi/xenon1t/minitrees/pax_v6.6.5 
     
Usage with MC:
    python lax_root_batch.py --run_number -1 --pax_version 6.6.5 --minitree_path output --filename Xenon1T_TPC_Rn222_00000_g4mc_G4_Sort_pax
"""
import argparse
import sys

import hax
from lax.lichens import sciencerun0

import root_pandas

parser = argparse.ArgumentParser(description="Create lichen ROOT files with lax")

parser.add_argument('--run_number', dest='RUN_NUMBER',
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

parser.add_argument('--output_path', dest='OUTPUT_PATH',
                    action='store', required=False, default='',
                    help='Name of output file (without .root)')

args = parser.parse_args(sys.argv[1:])

pax_version_policy = args.PAX_VERSION
run_number = args.RUN_NUMBER
minitree_names = ['Fundamentals', 'Basics', 'TotalProperties',
                  'Extended', 'TailCut', 'Proximity']
output_path = args.OUTPUT_PATH

LaxLichens = [sciencerun0.AllEnergy(), 
              sciencerun0.LowEnergyRn220(), 
              sciencerun0.LowEnergyAmBe(),
              sciencerun0.LowEnergyBackground()]
              
treename = 'tree'

# MC
if args.RUN_NUMBER < 0:
    
    # No run dependent sims yet
    pax_version_policy = 'loose'
    
    # Use filename instead of run number
    run_number = args.MINITREE_PATH+'/'+args.FILENAME
    
    # Remove meaningless variables
    minitree_names.remove('TailCut')
    minitree_names.remove('Proximity')

    # Remove meaningless cuts (HARDCODE WARNING)
    for cuts in LaxLichens:
        cuts.lichen_list.pop(6)  # DAQVeto
        cuts.lichen_list.pop(9)  # S2Tails

    treename += 'mc'
        
    if output_path is '':
        output_path = args.FILENAME+"_lax"

if output_path is '':
    output_path = "%d_lax" % run_number

# Initialize hax
kwargs = {'experiment': 'XENON1T',
          'pax_version_policy': pax_version_policy,
          'minitree_paths': ['.', args.MINITREE_PATH]
         }

# Be careful what you're doing here
if args.RUN_NUMBER < 0:
    kwargs['blinding_cut'] = 'run_number<=0'

hax.init(**kwargs)
        
df_all = hax.minitrees.load(run_number, minitree_names)

for cuts in LaxLichens:

    df_all = cuts.process(df_all)

    df_all.to_root(output_path+'.root', treename)
