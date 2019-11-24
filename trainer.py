import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split

df = pd.read_csv('./transformed_data_one_hot_genre-key.csv')[:]

classifier_c = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=100)
classifier_r = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100)

classifier = classifier_c

dropped_cols = ['popularity']
df_features = df.drop(columns=dropped_cols)
#Classifier
df_labels = [ 0 if pop < 66 else 1 for pop in df['popularity'] ]
#Multinomial
# df_labels = [int(round(pop/20)) for pop in df['popularity']]
#Regression
# df_labels = df['popularity']

X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=0)
classifier.fit(X_train, y_train)
print('Score: ', classifier.score(X_test, y_test))
for ft, fti in zip(df_features, classifier.feature_importances_ ):
    print(ft, '{:.4f}'.format(fti*100))