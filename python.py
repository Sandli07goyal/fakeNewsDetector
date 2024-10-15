import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
from matplotlib import pyplot as plt

# Load the dataset
df = pd.read_csv("fake_or_real_news.csv")
df = df.set_index("Unnamed: 0")  # Set index if necessary
y = df.label  # Target variable

# Prepare the feature variable
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# Vectorization
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names_out())
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names_out())

# Find differences between features
difference = set(count_df.columns) - set(tfidf_df.columns)
print("Features in Count Vectorizer not in TF-IDF Vectorizer:", difference)
print(count_df.equals(tfidf_df))


# Plot confusion matrix function
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Train and evaluate classifiers
def train_and_evaluate_models(tfidf_train, y_train, tfidf_test, y_test, count_train, count_test):
    classifiers = {
        'MultinomialNB (TF-IDF)': MultinomialNB(),
        'MultinomialNB (Count)': MultinomialNB(),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(max_iter=50)
    }

    for name, clf in classifiers.items():
        # Fit the model
        if 'Count' in name:
            clf.fit(count_train, y_train)
            pred = clf.predict(count_test)
        else:
            clf.fit(tfidf_train, y_train)
            pred = clf.predict(tfidf_test)

        # Calculate accuracy and confusion matrix
        score = metrics.accuracy_score(y_test, pred)
        print(f"{name} accuracy: {score:.3f}")
        cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
        plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
        plt.show()


# Call the function to train and evaluate models
train_and_evaluate_models(tfidf_train, y_train, tfidf_test, y_test, count_train, count_test)


# Define the function for most informative features
def most_informative_feature_for_binary_classification(vectorizer, classifier, n=30):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names_out()  # Use get_feature_names_out() for new versions
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    print("Most informative features for class 0:")
    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)

    print("\nMost informative features for class 1:")
    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)


# Train the final model for most informative features
linear_clf = PassiveAggressiveClassifier(max_iter=50)
linear_clf.fit(tfidf_train, y_train)

most_informative_feature_for_binary_classification(tfidf_vectorizer, linear_clf, n=30)
