import pandas as pd
import numpy as np

df = pd.read_csv('spam.csv', encoding='latin-1')

X = df['v2']
y = np.array(df['v1'])

#Data Visualization
import matplotlib.pyplot as plt
plt.xlabel('Label')
plt.title('Number of ham and spam messages')
unique, counts = np.unique(y, return_counts=True)
plt.bar(unique,counts)
plt.show()

#Cleaning of text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 5572):
    message = re.sub('[^a-zA-Z_0-9]', ' ', X[i])
    message = message.lower()
    message = message.split()
    ps = PorterStemmer()
    message = [ps.stem(word) for word in message if not word in set(stopwords.words('english'))]
    message = ' '.join(message)
    corpus.append(message)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 150000)
X = cv.fit_transform(corpus).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
acc = 0
y_pred = classifier.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        acc += 1
print(acc/len(y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
