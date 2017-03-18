"""Make ROOT file of all lax output
"""
import argparse
import sys

import hax
from lax.lichens import sciencerun0

import root_pandas

parser = argparse.ArgumentParser(description="Create lichen ROOT files with lax")

parser.add_argument('--run_number', dest='RUN_NUMBER',
                    action='store', required=True, type=int,
                    help='Run number to process')

parser.add_argument('--pax_version', dest='PAX_VERSION',
                    action='store', required=True,
                    help='pax version to process')

args = parser.parse_args(sys.argv[1:])

hax.init(experiment='XENON1T',
         pax_version_policy=args.PAX_VERSION,
         minitree_paths = ['.','/project2/lgrandi/xenon1t/minitrees/pax_v6.4.2'],
         #make_minitrees = False
)

DF = hax.minitrees.load(args.RUN_NUMBER,
                        ['Fundamentals', 'Basics', 'TotalProperties',
                         'Extended', 'Proximity', 'TailCut'],
)

OLD_COLUMNS = DF.columns

for cuts in [sciencerun0.AllEnergy(),
             sciencerun0.LowEnergy()]:
    df_temp = cuts.process(DF.copy())
    new_columns = [column for column in df_temp.columns if column not in OLD_COLUMNS]

    cuts_name = cuts.name()

    df_temp.loc[:, new_columns].to_root('lax_%s_%d.root' % (cuts_name,
                                                            args.RUN_NUMBER))

