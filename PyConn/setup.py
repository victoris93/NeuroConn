from setuptools import setup, find_packages

setup(
    name='PyConn',
    version='1.0.0',
    description='A BIDS toolbox for connectivity & gradient analyses.',
    author='Victoria Shevchenko',
    author_email='shevchenko682@gmail.com',
    packages=find_packages(),
    url='https://github.com/victoris93/PyConn',  # Added comma here
    install_requires=[
        'nilearn',
        'numpy',
        'pandas',
        'scikit-learn',
        'nibabel',
        'brainspace',
        'gdown',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    keywords='fmriprep, BIDS, connectivity, gradients, dispersion',
)