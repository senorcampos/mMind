"""Build a sentiment analysis / polarity model

Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess wether the opinion of the author is
positive or negative.

In this examples we will use a movie review dataset.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD
# modified by Michael Campos

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

import csv

import web

urls = (
  '/', 'index'
)

app = web.application(urls, globals())

render = web.template.render('templates/')

class index:
    def GET(self):
        #greeting = "Hello World"
        
        f = open('./movie-reviews-sentiment.tsv','r')
        cf = csv.reader(f, delimiter='\t')
        target_names = ['negative', 'positive']
    
        count = -1
        target = [];
        data = [];
        for w in cf:
            count = count+1
            if w[0] == 'negative':
                target.append(0)
            else:    
                target.append(1)
        
            data.append(w[1])
            
        greeting= str(len(data))    
        
        return render.index(greeting = greeting)

#if __name__ == "__main__":



if __name__ == "__main__":
    app.run()
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV

    # the training data folder must be passed as first argument
    #movie_reviews_data_folder = sys.argv[1]
    #dataset = load_files(movie_reviews_data_folder, shuffle=False)

    #f = open('/Users/michael/Desktop/IntelligentMachines/MetaMind/movie-reviews-sentiment.tsv','r')
    f = open('./movie-reviews-sentiment.tsv','r')
    cf = csv.reader(f, delimiter='\t')
    target_names = ['negative', 'positive']

    count = -1
    target = [];
    data = [];
    for w in cf:
        count = count+1
        if w[0] == 'negative':
            target.append(0)
        else:    
            target.append(1)
    
        data.append(w[1])

    print("n_samples: %d" % len(data))





    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        data, target, test_size=0.25, random_state=None)

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        #('clf', LinearSVC(C=1000)),
        ('clf',MultinomialNB()),
        #('clf',LogisticRegression()),
    ])

    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
    grid_search.fit(docs_train, y_train)

    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search
    print(grid_search.grid_scores_)

    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted
    y_predicted = grid_search.predict(docs_test)

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=target_names))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    #import matplotlib.pyplot as plt
    #plt.matshow(cm)
    #plt.show()

    # display some results
    # should be better for some classifiers than others
    # how do I call this thing from a JavaScript Web Application?

