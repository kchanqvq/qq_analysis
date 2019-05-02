from datetime import datetime
import numpy as np
import operator
result={}
with open("timestamps.txt","r") as infile:
    result=eval(infile.read())
nicknames={}
with open("names.txt","r") as infile:
    nicknames=eval(infile.read())
startdate = datetime(2019,2,6,21,10,10).timestamp()
enddate = datetime(2019,5,2,10,9,0).timestamp()
result = {k:list(filter(lambda x:x>startdate,v)) for k,v in result.items()}
def corr(id1,id2,binsize):
    rec1=result[id1]
    rec2=result[id2]
    x=np.arange(startdate,enddate,binsize)
    hist1,_=np.histogram(rec1,x)
    hist2,_=np.histogram(rec2,x)
    return np.sum(hist1*hist2),len(rec1),len(rec2)
def normcorr(id1,id2,binsize):
    vcorr,n1,n2=corr(id1,id2,binsize)
    return vcorr/n1/n2
def interests(id1,binsize):
    ids = result.keys()
    interestdict = {}
    for id2 in ids:
        if id1 != id2:
            interestdict[id2] = normcorr(id1,id2,binsize)
    return sorted(interestdict.items(), key=operator.itemgetter(1), reverse=True)
def matcorr(binsize):
    mat=[]
    labels=[]
    ids = result.keys()
    x=np.arange(startdate,enddate,binsize)
    for id1 in ids:
        labels.append(id1)
        mat.append(np.histogram(result[id1],x)[0])
    mat=np.array(mat)
    return labels,np.corrcoef(mat)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
font=FontProperties(fname="/usr/local/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/STHei.ttf", size=8)
from sklearn import cluster, covariance, manifold
binsize=60
ids=result.keys()
names=[]
variation = []
x=np.arange(startdate,enddate,binsize)
for id1 in ids:
    rec = result[id1]
    if len(rec) > 100:
        #names.append(nicknames[id1])
        names.append(nicknames[id1]+"("+str(id1)+")")
        variation.append(np.histogram(rec,x)[0])
variation=np.array(variation)
# #############################################################################
# Learn a graphical structure from the correlations
edge_model = covariance.GraphicalLassoCV(cv=5)

# standardize the time series: using correlations rather than covariance
# is more efficient for structure recovery
X = variation.copy().T.astype("f")
stdX = X.std(axis=0)

edge_model.fit(X)

# #############################################################################
# Cluster using affinity propagation

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()
'''
for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))
'''
# #############################################################################
# Find a low-dimension embedding for visualization: find the best position of
# the nodes (the stocks) on a 2D plane

# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

embedding = node_position_model.fit_transform(X.T).T
# #############################################################################
# Visualization
plt.figure(1, facecolor='w', figsize=(100, 100))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100000* d ** 2, c=labels,
            cmap=plt.cm.nipy_spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
# a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.log(np.abs(partial_correlations[non_zero]))
values -= values.min()
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .8*values.max()))
lc.set_array(values)
lc.set_linewidths(5 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(names, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .001
    else:
        horizontalalignment = 'right'
        x = x - .001
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .001
    else:
        verticalalignment = 'top'
        y = y - .001
    plt.text(x, y, name, size=5,fontproperties=font,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                       alpha=.6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

#plt.show()
plt.savefig("map2019-2.pdf")
