from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tom-iag',
    version='0.1',
    description='IAG telescopes facility module for the TOM Toolkit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/thusser/tom_iag',
    author='Tim-Oliver Husser',
    author_email='thusser@uni-goettingen.de',
    packages=find_packages(),
    install_requires=[
        'tomtoolkit>=2.0'
    ]
)
