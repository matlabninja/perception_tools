from setuptools import setup, find_packages

VERSION = '0.2'

setup_info = dict(
    name='perceptionTools',
    version = VERSION,
    author = 'Andrew Lingg',
    description = 'Collection of AI/ML, computer vision, and signal processing algorithms',
    license = 'MIT',
    packages = ['perceptionTools'] + ['perceptionTools.' + pkg for pkg in find_packages('perceptionTools')],
    install_requires = [
        'numpy>=1.24.3',
        'scipy>=1.10.1',
        'torch>=2.0.1',
        'torchvision>=0.15.2',
    ]
)

setup(**setup_info)