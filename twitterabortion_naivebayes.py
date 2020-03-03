# Main libraries
import pandas as pd
import numpy as np
import nltk

# NLTK specific libraries
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

# Scikit Learn specific libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.cross_validation import KFold, cross_val_score

# Visualization libraries
import matplotlib.pyplot as plt
from wordcloud import WordCloud



# 1.1 Read in the raw_tweets and ground_truth data
raw_tweets_df = pd.read_table('AbortionTwitterData/raw_tweets.txt',  sep='\t')
print raw_tweets_df.head()
print "raw_tweets size is", raw_tweets_df.shape[0]

ground_truth_df = pd.read_table('AbortionTwitterData/ground_truth.txt',  sep='\t')
print ground_truth_df.head()
print "groundturth size is", ground_truth_df.shape[0]

# 1.2 Take labels from ground_truth and assign them to the corresponding raw_tweets
raw_tweets_df['Label'] = "Neutral"
print raw_tweets_df.head()
for row in range(ground_truth_df.shape[0]):
    raw_tweets_df.loc[row, 'Label'] = ground_truth_df.loc[row, 'Label']

# 2.2 Preprocessing TO LOWERCASE - convert all words in tweets to lowercase
raw_tweets_df['Tweet Text'] = raw_tweets_df['Tweet Text'].str.lower()
# df['url'] = df['url'].str.lower()

# 2.3 Preprocessing STOPWORDS - eliminate stopwords in the text (ie. 'a', 'the', etc.)
stop_words = set(stopwords.words("english")) 
raw_tweets_df['Tweet Text'].apply(lambda x: [item for item in x if item not in stop_words])

print "df with stopwords removed is"
print raw_tweets_df

# 2.4 Preprocessing PUNCTUATION - remove punctuation marks
raw_tweets_df['Tweet Text'] = raw_tweets_df['Tweet Text'].str.replace('[^\\w\\s]', '')

# 2.5 Preprocessing TOKENIZER
raw_tweets_df['Tweet Text'] = raw_tweets_df['Tweet Text'].apply(nltk.word_tokenize)  

# # 2.6 Preprocessing STEMMING - reduce words to common stems (ie. flying, fly, flyer get reduced to fly)
# stemmer = PorterStemmer()
# raw_tweets_df['Tweet Text'] = raw_tweets_df['Tweet Text'].apply(lambda x: [stemmer.stem(y) for y in x])

# 2.7 Preprocessing  This converts the list of words into space-separated strings
raw_tweets_df['Tweet Text'] = raw_tweets_df['Tweet Text'].apply(lambda x: ' '.join(x))
count_vect = CountVectorizer()  
counts = count_vect.fit_transform(raw_tweets_df['Tweet Text']) 

# 3 Use Term Frequency Inverse Document Frequency -- tf-idf to get a more accurate word count
transformer = TfidfTransformer().fit(counts)
counts = transformer.transform(counts) 

# 4.1 To begin training, we first split our CSV data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(counts, raw_tweets_df['Label'], test_size=0.2, random_state=69)

# 4.2 We initialize the model we will use as our classifier -- Naive Bayes Classifier, and train it with training data
model = MultinomialNB().fit(X_train, y_train)

# 5. Evaluate the performance of the Naive Bayes classifier on this data
predicted = model.predict(X_test)

print ""
print "Accuracy is"
accuracy = np.mean(predicted == y_test)
print accuracy

print ""
print "F1 Score is"
f1 = f1_score(y_test, predicted, average="macro")
print f1

print ""
print "Precision Score is"
precision = precision_score(y_test, predicted, average="macro")
print precision


print ""
print "Recall Score is"
recall = recall_score(y_test, predicted, average="macro")
print recall
print ""


against_abortion = ' '.join(list(raw_tweets_df[raw_tweets_df['Label'] == 'Neutral']['Tweet Text']))
against_abortion_wc = WordCloud(width = 512,height = 512).generate(against_abortion)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(against_abortion_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig('Neutral.png')

against_abortion = ' '.join(list(raw_tweets_df[raw_tweets_df['Label'] == 'For Abortion']['Tweet Text']))
against_abortion_wc = WordCloud(width = 512,height = 512).generate(against_abortion)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(against_abortion_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig('ForAbortion.png')

against_abortion = ' '.join(list(raw_tweets_df[raw_tweets_df['Label'] == 'Against Abortion']['Tweet Text']))
against_abortion_wc = WordCloud(width = 512,height = 512).generate(against_abortion)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(against_abortion_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig('AgainstAbortion.png')

k_fold = KFold(len(y_train), n_folds=10, shuffle=True, random_state=0)
clf = model
print "k-fold cross validation score for k=10"
print cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1)

