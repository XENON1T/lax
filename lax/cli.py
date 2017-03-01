"""lax client

Just prints out version information.
"""
# -*- coding: utf-8 -*-

import lax
import click


@click.command()
def main(args=None):
    """Console script for lax"""
    click.echo('lax version: %s\n' % lax.__version__)

    for cut_set in [lax.lichens.sciencerun0.AllEnergy(),
                    lax.lichens.sciencerun0.LowEnergy()]:
        click.echo(cut_set.name())
        for each in cut_set.lichen_list:
            click.echo('\t%s version %s' % (each.name(),
                                          each.version))


if __name__ == "__main__":
    main()
