import pandas as pd
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('./transformed_data_one_hot_genre-key.csv')[:]
# df = pd.read_csv('./transformed_data_all_std.csv')

classifier_c = RandomForestClassifier(max_depth=100, random_state=0, n_estimators=200)
classifier_r = RandomForestRegressor(max_depth=100, random_state=0, n_estimators=200)

dropped_cols = ['popularity']
features_df = df.drop(columns=dropped_cols)

def do_binomial(features_df, threshold):
    classifier = classifier_c
    print('*** THRESH {} ***'.format(threshold))
    df_labels = [ 0 if pop < threshold else 1 for pop in df['popularity'] ]
    print("% of Popular songs", sum(df_labels)/len(df.index))
    X_train, X_test, y_train, y_test = train_test_split(features_df, df_labels, test_size=0.2, random_state=0)
    classifier.fit(X_train, y_train)
    print('Thresh: {}, Score: '.format(threshold), classifier.score(X_test, y_test))
    y_pred = classifier.predict(X_test)
    print('Score:',classifier.score(X_test, y_test))
    print('F1', metrics.f1_score(y_test, y_pred))

def do_multiclass(features_df, n_classes):
    classifier = classifier_c
    scalar = 100/n_classes
    df_labels = [int(round(pop/scalar)) for pop in df['popularity']]
    X_train, X_test, y_train, y_test = train_test_split(features_df, df_labels, test_size=0.2, random_state=0)
    classifier.fit(X_train, y_train)
    print('Score:',classifier.score(X_test, y_test))

def do_regression(features_df):
    classifier = classifier_r
    df_labels = df['popularity']
    X_train, X_test, y_train, y_test = train_test_split(features_df, df_labels, test_size=0.2, random_state=0)
    classifier.fit(X_train, y_train)
    print('Score:',classifier.score(X_test, y_test))


print('*** BINOMIAL ***')
do_binomial(features_df, 44)
print('*** BINOMIAL ***')
do_binomial(features_df, 70)
n_classes = 3
print('*** MULTICLASS ({}) *** '.format(n_classes))
do_multiclass(features_df, n_classes)
n_classes = 5
print('*** MULTICLASS ({}) *** '.format(n_classes))
do_multiclass(features_df, n_classes)
n_classes = 10
print('*** MULTICLASS ({}) *** '.format(n_classes))
do_multiclass(features_df, n_classes)
print('*** REGRESSION ***')
do_regression(features_df)

