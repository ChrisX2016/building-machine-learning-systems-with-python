import os
import scipy as sp
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

DIR = r"content"
posts = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]

import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')


# class StemmedCountVectorizer(CountVectorizer):
#     def build_analyzer(self):
#         analyzer = super(StemmedCountVectorizer, self).build_analyzer()
#         return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
#
# vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer,self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english')
# vectorizer = CountVectorizer(min_df=1)
X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape
# print(vectorizer.get_feature_names())


new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])


def dist_norm(v1, v2):
    # print(v1.toarray())
    # print(v2.toarray())
    v1_norm = v1/sp.linalg.norm(v1.toarray())
    v2_norm = v2/sp.linalg.norm(v2.toarray())
    delta = v1_norm - v2_norm
    return sp.linalg.norm(delta.toarray())

best_doc = None
best_dist = 1000
best_i = None
for i in range(0, num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_norm(post_vec, new_post_vec)
    print("=== Post %i with dist=%.2f: %s" % (i, d, post))
    if d < best_dist:
        best_dist = d
        best_i = i
print("Best post is %i with dist=%.2f"%(best_i, best_dist))

from sklearn.cluster import KMeans

num_claster = 2
km = KMeans(n_clusters=num_claster, init='random', n_init=1, verbose=1)
km.fit(X_train.toarray())
print(km.labels_)
