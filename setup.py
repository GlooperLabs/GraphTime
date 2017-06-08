from distutils.core import setup

current_version = '0.1.1'

setup(
    name='graphtime',
    packages=['graphtime'],
    version=current_version,
    description='Dynamic Graphical Model Estimation',
    author='Glooper Limited',
    author_email='alex.gibberd@glooper.io, alex.immer@glooper.io',
    url='https://github.com/GlooperLabs/GraphTime',
    download_url='https://github.com/AlexImmer/GlooperLabs/GraphTime/v'
                + current_version + '.tar.gz',
    keywords=['dynamic', 'graphical models', 'time series', 'non-stationary'],
    classifiers=[]
)
