run_number = 3956
pax_version = '6.4.2'

import hax

hax.init(experiment='XENON1T',
         pax_version_policy = pax_version)

df = hax.minitrees.load(run_number,
                        [ 'Fundamentals', 'Basics', 'TotalProperties'])

from lax.lichens import sciencerun0
all_cuts = sciencerun0.AllCuts()
low_energy_cuts = sciencerun0.LowEnergyCuts()

old_columns = df.columns

import root_pandas
for cuts in [all_cuts, low_energy_cuts]:
    df_temp = cuts.process(df.copy())
    new_columns = [column for column in df_temp.columns if column not in old_columns]
    
    cuts_name = all_cuts.__class__.__name__
    
    df_temp.loc[:,new_columns].to_root('lax_%s_%d.root' % (cuts_name,
                                        run_number))
