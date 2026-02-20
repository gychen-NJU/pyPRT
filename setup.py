import os
from setuptools import setup, find_packages

setup(
    name='pyprt',
    version='0.1.0',
    author='Guoyin Chen',
    author_email='gychen@smail.nju.edu.cn',
    description='A Python library for synthesis Stokes spectral',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'pyprt': [
            'data/*.csv', 'data/*.txt', 'data/*.json','data/*.pkl',
            'data/OPtabs/*.csv',
            'data/model_atmosphere/*.txt'
            ],
    },
    install_requires=[
        'numpy>=1.18.0',
        'torch>=2.5.0',
        'scipy>=1.15.0',
        'pandas>=2.2.0',
        'periodictable>=2.0.0',
        'setuptools<=82.0.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)