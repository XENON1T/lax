#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    'numpy',
    'seaborn>=0.7.1',
    'matplotlib',
    'scipy'
]

test_requirements = [
    'nose',
]

setup(
    name='lax',
    version='0.8.6',
    description="Package for standardizing event selections on hax minitrees.",
    long_description=readme + '\n\n' + history,
    author="Christopher Tunnell",
    author_email='tunnell@uchicago.edu',
    url='https://github.com/tunnell/lax',
    packages=[
        'lax',
        'lax.lichens'
    ],
    package_dir={'lax':
                 'lax'},
    entry_points={
        'console_scripts': [
            'lax=lax.cli:main'
        ]
    },
    package_data={'lax': ['data/*.*']},
    include_package_data=True,
    install_requires=requirements,
    license="GNU General Public License v3",
    zip_safe=False,
    keywords='lax',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
