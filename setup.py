#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Melvin Alvarez-Caradu",
    author_email='melvin.caradu@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Monte-Carlo Simulation of the so-called 2048 game using numba as input of Markov-Chain-Monte-Carlo algorithm",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='magic_number_2048_monte_carlo',
    name='magic_number_2048_monte_carlo',
    packages=find_packages(include=['magic_number_2048_monte_carlo', 'magic_number_2048_monte_carlo.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ThotDjehuty/magic_number_2048_monte_carlo',
    version='0.1.0',
    zip_safe=False,
)
