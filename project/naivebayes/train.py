from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import compute_sample_weight

train_df = pd.read_csv('data/civil_comments/train.csv')
test_df = pd.read_csv('data/civil_comments/train.csv')

train_df['toxicity'] = (train_df['toxicity'] > 0.5) * 1
test_df['toxicity'] = (test_df['toxicity'] > 0.5) * 1

samples_weight = compute_sample_weight('balanced', train_df['toxicity'])
print(train_df.head(5))

count_vect = CountVectorizer(stop_words='english')
tfidf_transformer = TfidfTransformer()

train_df['comment_text'] = train_df['comment_text'].astype('U')
test_df['comment_text'] = test_df['comment_text'].astype('U')

X_train_counts = count_vect.fit_transform(train_df['comment_text'].values)
print(X_train_counts.shape)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, train_df['toxicity'], sample_weight=samples_weight)

X_new_counts = count_vect.transform(test_df['comment_text'])
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
print(Counter(train_df['toxicity']))
print(accuracy_score(predicted, test_df['toxicity']))
print(f1_score(predicted, test_df['toxicity']))
print(predicted.shape)

print(classification_report(predicted, test_df['toxicity']))

# Dla progu 0.5:
# (1799564, 318216)
# Counter({0: 1693559, 1: 106005})
# 0.7994892096085496
# 0.34576178992983153
# (1799564,)
#               precision    recall  f1-score   support
#
#            0       0.79      0.99      0.88   1354039
#            1       0.90      0.21      0.35    445525
#
#     accuracy                           0.80   1799564
#    macro avg       0.85      0.60      0.61   1799564
# weighted avg       0.82      0.80      0.75   1799564
