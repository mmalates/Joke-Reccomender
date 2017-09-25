from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk.data
import numpy as np
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd

path2 = 'data/ratings.dat'
path = 'data/jokes.dat'
with open(path) as f:
    soup = BeautifulSoup(f, 'html.parser')


jokes = soup.findAll('p')

stuff = ['''\n''', '''</p>''', '''\r''', '''<br/>''', '''<p>''', '''<br>''']
joke_lst = []
for joke in jokes:
    one_joke = ''.join(str(joke.encode()))
    for char in stuff:
        one_joke = one_joke.replace(char, ' ')
    joke_lst.append(one_joke)


jf = pd.DataFrame(columns=['joke_id', 'contents', 'cluster_id'])
jf['joke_id'] = range(1, len(jokes) + 1, 1)
jf['contents'] = joke_lst
jf['cluster_id'] = 0

vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(jf.contents).toarray()
words = vectorizer.get_feature_names()

from sklearn.decomposition import NMF
model = NMF(n_components=10, init='random', random_state=0)
W = model.fit_transform(vectors)
H = model.components_

vect_nmf = np.dot(W, H)
vect_nmf
vectors.shape
vect_nmf.shape
kmeans = KMeans(n_clusters=10)
kmeans.fit(vect_nmf)
assigned_cluster = kmeans.transform(vect_nmf).argmin(axis=1)
lst = np.array(len(jf.contents))

for i in range(kmeans.n_clusters):
    cluster = np.arange(0, vectors.shape[0])[assigned_cluster == i]
    for joke in cluster:
        mask = jf['joke_id'] == joke
        jf.loc[mask, 'cluster_id'] = i
        print i
        print "    {} {}".format(jf.loc[joke]['joke_id'], jf.loc[joke]['contents'])


jf_side = jf.drop(['contents'], axis=1)

jf_side.to_csv('side_features_10.csv', index=False)

#
#
# from numpy.linalg import svd, eig
#
# df = pd.read_table(path2)
# df.rating = df.rating+10
# df.rating.head()
# df.head()
# dfpiv = df.pivot(index='user_id', columns='joke_id', values='rating')
# dfpiv = dfpiv.fillna(0)
# dfpiv.head()
# u, s, v = svd(dfpiv, full_matrices=0)
# power = s * s
#
# def plotcurve():
#     fig, ax = plt.subplots()
#     ax.plot(power[1:20])
#     plt.xlim([0, 20])
#     plt.ylim([])
#     plt.show()
#
# tot_power = []
# for i in xrange(len(power)):
#     tot_power.append(sum(power[:i]) / sum(power))
#
# def plotpower():
#     fig, ax = plt.subplots()
#     ax.plot(tot_power)
#     plt.xlim([1,100])
#     ax.hlines(0.9, 0, 100)
#     plt.show()
#
# plotpower()
