"""
https://datascienceplus.com/multi-class-text-classification-with-scikit-learn/
https://scikit-learn.org/stable/modules/feature_extraction.html

try n-gram tfidf vectorizer -- 3-5 gram
try document-level-- randomly sample some documents without warnings
look at most important features

IR for sampling-- index by links, index by backlinks, search by title


removing stopwords? other categories of words? remove countries/nationalities as features?

run current model on whole documents
run current model on first <400ish> words of article, see what it does

for NN: train on document level first, then sentence level



throw away all nouns and verbs?
work on information retrieval section

"""
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

peacock_filename = "clean-short-nostop-peacockterms.txt"
normal_filename = "clean-short-nostop-nonpeacock_random_append.txt"

df = [[line.strip(), 1] for line in open(peacock_filename).readlines()]
df.extend([[line.strip(), 0] for line in open(normal_filename).readlines()])
random.shuffle(df)
df = np.array(df)
print(np.shape(df))
print(df[:, 0])


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,5))
features = tfidf.fit_transform(df[:, 0]).toarray()
labels = df[0,:]
print(features.shape)

"""from sklearn.feature_selection import chi2
import numpy as np
N = 2
for category_id in sorted(df[:,0]):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigramscategory_to_id:\n. {}".format('\n. '.join(bigrams[-N:])))"""


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = train_test_split(df[:,0], df[:,1], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = LinearSVC().fit(X_train_tfidf, y_train)
print(tfidf_transformer.transform(count_vect.transform(X_test)))
y_pred = clf.predict(tfidf_transformer.transform(count_vect.transform(X_test)))


print(X_test[0])
print(clf.predict(count_vect.transform([X_test[0]])))
print(y_test[0])
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=[0,1], yticklabels=[0,1])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
def plot_coefficients(classifier, feature_names, top_features=20):
 coef = classifier.coef_.ravel()
 top_positive_coefficients = np.argsort(coef)[-top_features:]
 top_negative_coefficients = np.argsort(coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 # create plot
 plt.figure(figsize=(15, 5))
 plt.bar(np.arange(2 * top_features), coef[top_coefficients])
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(1, 1 + 2 * top_features),
            feature_names[top_coefficients], rotation=60)
 plt.show()

plot_coefficients(clf, count_vect.get_feature_names())

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
labels=["normal", "peacock"]
CV = 2
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
df_counts = count_vect.fit_transform(df[:,0])
df_tfidf = tfidf_transformer.fit_transform(df_counts)

for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, df_tfidf, df[:,1], scoring='f1_micro')
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


