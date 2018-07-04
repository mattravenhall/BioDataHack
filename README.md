# BioDataHack
Team J-SCAMP's project for [BioDataHack](https://www.sanger.ac.uk/innovations/hackathon2018). Repurposing drugs and reclassifying diseases through unsupervised machine learning.

## Prerequisites
Versions used for development:
- python (v3.4)
- numpy (v1.11.3)
- pandas (v0.19.2)
- matplotlib (v2.0.0)
- sklearn (v0.18.1)
- opentargets (v3.1.0) (http://opentargets.readthedocs.io/en/stable/)

## Running
Running *main.py* will pull down a disease-gene data from OpenTargets (unless that dataset is in the current working directory) and return two plots. Those include 'pca.png', which shows Principal Components 1 and 2 where diseases are red points and drugs are blue, and 'kmeans.png', where the four most likely clusters are colourised. The drug-gene (& disease-gene) dataset should present in the current working directory.

## Parameters
By default *main.py* will perform dimension reduction with Principal Component Analysis, but Singular-Value Decomposition and t-SNE can be performed if *clusterType* is set to 'gene-svd' or 'gene-tsne'. Nb. these options will require editing *main.py*. Similarly, if disease-gene information needs to be pulled from OpenTargets the script will pull down 500 diseases by default, with more or less being pulled by changing the *maxDiseases* variable.
