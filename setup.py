#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
from setuptools import setup, find_packages


with open('README.rst') as readme_file:
    readme = readme_file.read()

def get_version():
    _globals = {}
    with open(os.path.join("autodqm", "version.py")) as version_file:
        exec(version_file.read(), _globals)
    return _globals["__version__"]

requirements = ['certifi=2018.4.16', 'chardet==3.0.4', 'idna==2.7', 'lxml==4.2.3',
                'requests-futures==0.9.7', 'requests==2.19.1', 'tqdm==4.23.4',
                'urllib3==1.2.3', 'python_version>=2.6']

setup(
    author="Rob White",
    author_email='robert.stephen.white@cern.ch',
    classifiers=[
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Statistical analysis toolkit for DQM of histograms",
    install_requires=requirements,
    long_description=readme,  # + '\n\n' + history,
    include_package_data=True,
    keywords=['ROOT', 'pandas', 'analysis', 'particle physics', 'HEP', 'F.A.S.T'],
    name='AutoDQM',
    url='https://github.com/GluonicPenguin/AutoDQM',
    zip_safe=True,
)
