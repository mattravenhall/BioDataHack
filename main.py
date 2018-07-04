print('Importing libraries.')
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from opentargets import OpenTargetsClient

removeDrugSpecific = False
removeDiseaseSpecific = False

maxDiseases = 500
clusterType = 'gene-pca'

# Functions
def getGeneList(disease):
	return(list(set([x['target']['gene_info']['symbol'] for x in ot.get_associations_for_disease(disease)])))

# KEGG Disease IDs
print('Reading in disease and drug IDs.')
diseases = pd.read_table('id_dis.txt',header=None,names=['ID','Name'])
drugs = pd.read_table('id_drug.txt',header=None,names=['ID','Name'])

# Reading in drug gene relations
drugGenes = pd.read_csv('gdi.csv', index_col=0) #pd.read_csv('drug_gene_data.csv', index_col=0)
if removeDrugSpecific:
	drugGenes = drugGenes.loc[drugGenes.sum(axis=1) > 5,:] # Remove less informative rows

# Pull disease names
diseases = list(diseases.Name.str.lower())[:maxDiseases] #diseases = ['aids', 'liver cancer', 'breast cancer', 'liver cancer', 'prostate cancer']

# Initialise client/s
print('Initialising OT client.')
ot = OpenTargetsClient()

if not os.path.isfile('disease_gene_data.csv'):
	# Create gene matrix
	geneLists = {}
	allTargets = set()

	# Creating gene matrix
	for disease in diseases:
		print('Pulling {0} of {1} diseases...'.format(diseases.index(disease)+1, len(diseases)))	
		try:
			geneLists[disease] = getGeneList(disease)
			allTargets = allTargets.union(geneLists[disease])
		except:
			pass
	allTargets = list(allTargets)

	geneDF = {}
	for disease in diseases:
		print('Converting {0} of {1} diseases...'.format(diseases.index(disease)+1, len(diseases)))
		geneDF[disease] = [int(x in geneLists[disease]) for x in allTargets]

	df = pd.DataFrame.from_dict(geneDF).transpose()
	df.columns = allTargets
	df.drop(df.index[df.sum(axis=1) / df.shape[1] >= 0.8], inplace=True)
	df.drop(['chronic myeloid leukemia (cml)', 'acute myeloid leukemia (aml)'], inplace=True) # WARNING: Manual af
	df.to_csv('disease_gene_data.csv')
else:
	df = pd.read_csv('disease_gene_data.csv', index_col=0)

print('Combining drug and disease data.')
if removeDiseaseSpecific:
	df = df.loc[df.sum(axis=1) > 5,:] # Remove less informative rows
df = df.append(drugGenes).dropna(axis=1)
#df = df.loc[df.sum(axis=1) > 5,:] # Remove less informative rows
numDisease = sum([x not in drugGenes.index for x in df.index])
cols = (['red'] * numDisease) + (['blue'] * (df.shape[0]-numDisease))

# Perform PCA/SVD
if clusterType == 'gene-tsne':
	print('Performing t-SNE.')
	from sklearn.manifold import TSNE

	# Perform t-SNE
	dfx = TSNE(n_components=5).fit_transform(df)

	xvals = [x[0] for x in dfx]
	yvals = [x[1] for x in dfx]

	plt.scatter(xvals, yvals, c=cols)
	plt.savefig('tsne.png')
elif clusterType == 'gene-svd':
	print('Performing SVD.')
	from sklearn.decomposition import TruncatedSVD

	# Perform SVD
	dfx = TruncatedSVD(n_components=5, n_iter=7, random_state=42).fit_transform(df)

	xvals = [x[0] for x in dfx]
	yvals = [x[1] for x in dfx]

	plt.scatter(xvals, yvals, c=cols)
	plt.savefig('svd.png')
elif clusterType == 'gene-pca':
	print('Performing gene PCA.')
	from sklearn.decomposition import PCA

	# Perform PCA (gene-focused)
	dfx = PCA(n_components=30).fit_transform(df)

	xvals = [x[0] for x in dfx]
	yvals = [x[1] for x in dfx]

	# Visualise
	plt.scatter(xvals, yvals, c=cols)
	plt.savefig('pca.png')
elif clusterType == 'drug-pca':
	print('Performing drug-disease PCA.')
	from sklearn.decomposition import PCA
	# Perform PCA (drug-focused)
	dr_ds = pd.read_table('dr_ds.mat',sep=' ')

	dr_dsX = PCA(n_components=30).fit_transform(dr_ds)

	xvals = [x[0] for x in dr_dsX]
	yvals = [x[1] for x in dr_dsX]

	candidateIndexes = [list(dat.index).index('ds:H000{}'.format(x)) for x in range(13,33) if 'ds:H000{}'.format(x) in dat.index]
	cols = ['black' if index not in candidateIndexes else 'red' for index in range(len(xvals))]
	plt.scatter(xvals, yvals, c=cols)
	plt.savefig('pca.png')
else:
	print('clusterType {} not recognised'.format(clusterType))
	sys.exit()

# Perform and plot k-means
print('Performing k-means clustering.')
reduced_data = PCA(n_components=2).fit_transform(df)
kmeans = KMeans(init='k-means++', n_clusters=4, n_init=10)
kmeans.fit(reduced_data)

print('Plotting k-means clusters.')
h = .02
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cols)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.savefig('kmeans.png')

# import pdb; pdb.set_trace()
# # sys.exit()
# # Component dissection
# pca = PCA(n_components=30).fit(df)
# pd.DataFrame(pca.components_,columns=df.columns)