'''from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

file='idf.txt'

vectorizer = TfidfVectorizer(input='file', use_idf=True, lowercase=True,
                             analyzer='word', ngram_range=(1, 1),
                             stop_words=None)
tfidf_matrix = vectorizer.fit_transform(vectorizer)
print(tfidf_matrix)
'''
from sklearn.feature_extraction.text import TfidfVectorizer

dataset = ["Load up on guns, bring your friends",
           "It's fun to lose and to pretend",
           "She's over-bored and self-assured",
           "Oh no, I know a dirty word",
           "Hello"]

vectorizer = TfidfVectorizer()
weighted_matrix = vectorizer.fit_transform(dataset)
terms = vectorizer.get_feature_names()
print(terms)
print(weighted_matrix)