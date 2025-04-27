from setuptools import setup, find_packages

# Carica il contenuto del README e del CHANGELOG
with open("README.md", "r", encoding="utf-8") as readme_file:
    readme_content = readme_file.read()

with open("CHANGELOG.md", "r", encoding="utf-8") as changelog_file:
    changelog_content = changelog_file.read()

# Combina il contenuto dei due file
long_description = readme_content + "\n\n" + "## Changelog\n\n" + changelog_content

__version__ = '1.0.7'

setup(
    name='LabToolbox',
    version=__version__,
    packages=find_packages(),
    long_description=long_description,  # Combina entrambi i contenuti
    long_description_content_type="text/markdown",  # Se usi README.md
    install_requires=[
        'numpy', 'scipy', 'matplotlib.pyplot', 'statsmodels.api', 'math', 'lmfit', 'corner', 'emcee'
    ],
    author='Giuseppe Sorrentino',
    author_email='sorrentinogiuse@icloud.com',
    description="LabToolbox è una raccolta di strumenti per l'analisi e l'elaborazione di dati sperimentali in ambito scientifico.",
    url="https://github.com/giusesorrentino/LabToolbox",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    license='MIT',  # Aggiungi questa linea
)