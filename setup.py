# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-13 09:40:02
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-01 16:13:33

import os
from setuptools import setup


def readme():
    with open('README.md', encoding="utf8") as f:
        return f.read()

def getFiles(path):
    return [f'{path}/{x}' for x in os.listdir(path)]


setup(
    name='PySONIC',
    version='1.0',
    description='Python implementation of the **multi-Scale Optimized Neuronal Intramembrane \
               Cavitation** (SONIC) model to compute individual neural responses to acoustic \
               stimuli, as predicted by the *intramembrane cavitation* hypothesis.',
    long_description=readme(),
    url='https://iopscience.iop.org/article/10.1088/1741-2552/ab1685',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    keywords=('SONIC NICE acoustic ultrasound ultrasonic neuromodulation neurostimulation excitation\
             computational model intramembrane cavitation'),
    author='Théo Lemaire',
    author_email='theo.lemaire@epfl.ch',
    license='MIT',
    packages=['PySONIC'],
    scripts=getFiles('scripts') + getFiles('tests'),
    install_requires=[
        'numpy>=1.10',
        'scipy>=0.17',
        'matplotlib>=2'
        'pandas>=0.22.0',
        'colorlog>=3.0.1',
        'tqdm>=4.3',
        'lockfile>=0.1.2',
        'multiprocess>=0.70',
        'pushbullet.py>=0.11.0'
    ],
    zip_safe=False
)
