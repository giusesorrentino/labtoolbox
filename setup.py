from setuptools import setup, find_packages

with open('CHANGELOG.md', 'r') as f:
    long_description = f.read()

__version__ = '1.0.7'

setup(
    name='LabToolbox',
    version=__version__,
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy', 'scipy', 'matplotlib.pyplot', 'statsmodels.api', 'math', 'lmfit', 'corner', 'emcee'
    ],
    author='Giuseppe Sorrentino',
    author_email='sorrentinogiuse@icloud.com',
    description="LabToolbox è una raccolta di strumenti per l'analisi e l'elaborazione di dati sperimentali in ambito scientifico.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # url='https://github.com/tuo-username/tuo-pacchetto',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    license='MIT',  # Aggiungi questa linea
)