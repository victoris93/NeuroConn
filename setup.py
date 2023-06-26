from setuptools import setup, find_packages
import os

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='NeuroConn',
    version='0.1.0a6',
    description='A BIDS toolbox for connectivity & gradient analyses.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Victoria Shevchenko',
    author_email='shevchenko682@gmail.com',
    python_requires='>=3.6',
    packages=find_packages(),
    url='https://github.com/victoris93/NeuroConn',
    install_requires=[
        'nilearn',
        'numpy',
        'pandas',
        'scikit-learn',
        'nibabel',
        'brainspace',
        'gdown',
        'fmriprep-docker', 
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    include_package_data=True,
    package_data={'': [os.path.join(os.path.dirname(__file__), 'NeuroConn', 'gradient', 'margulies_grads_schaefer1000.npy'),
                       os.path.join(os.path.dirname(__file__), 'NeuroConn', 'gradient', 'hcp_grads_schaefer1000_pearson_95th.npy'),
                       os.path.join(os.path.dirname(__file__), 'NeuroConn', 'gradient', 'hcp_grads_schaefer1000_pearson_95th.npy')]},
    keywords='fmriprep, BIDS, connectivity, gradients, dispersion',
)