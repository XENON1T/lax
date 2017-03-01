"""Make ROOT file of all lax output
"""
import hax
from lax.lichens import sciencerun0

RUN_NUMBER = 3956
PAX_VERSION = '6.4.2'

hax.init(experiment='XENON1T',
         pax_version_policy=PAX_VERSION)

DF = hax.minitrees.load(RUN_NUMBER,
                        ['Fundamentals', 'Basics', 'TotalProperties'])

OLD_COLUMNS = DF.columns

for cuts in [sciencerun0.AllEnergy(),
             sciencerun0.LowEnergy()]:
    df_temp = cuts.process(DF.copy())
    new_columns = [column for column in df_temp.columns if column not in OLD_COLUMNS]

    cuts_name = cuts.name()

    df_temp.loc[:, new_columns].to_root('lax_%s_%d.root' % (cuts_name,
                                                            RUN_NUMBER))
