#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-13 09:40:02
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-09-12 17:18:11

from setuptools import setup


def readme():
    with open('README.md', encoding="utf8") as f:
        return f.read()


setup(name='PointNICE',
      version='1.0',
      description='A Python framework to predict the electrical response of various neuron types\
                   to ultrasonic stimulation, according to the Neuronal Intramembrane Cavitation\
                   Excitation (NICE) model. The framework couples an optimized implementation of\
                   the Bilayer Sonophore (BLS) model with Hodgkin-Huxley "point-neuron" models.',
      long_description=readme(),
      url='???',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      keywords=('ultrasound ultrasonic neuromodulation neurostimulation excitation\
                 biophysical model intramembrane cavitation NICE'),
      author='Théo Lemaire',
      author_email='theo.lemaire@epfl.ch',
      license='MIT',
      packages=['PointNICE'],
      scripts=['sim/ESTIM_run.py', 'sim/ASTIM_run.py'],
      install_requires=[
          'numpy>=1.10',
          'scipy>=0.17',
          'matplotlib>=2',
          'openpyxl>=2.4',
          'pyyaml>=3.11',
          'pycallgraph>=1.0.1'
      ],
      zip_safe=False)
