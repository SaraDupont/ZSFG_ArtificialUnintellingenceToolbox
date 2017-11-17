"""
Created on Wed Oct 11 11:13:39 2017

@author: davidbmccoy
"""
# Natural Language Processing to Detect Hedging in Radiology Reports 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from __future__ import division
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import operator
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import *
import pickle
import os
from time import sleep
import sys
import io
import json

report_path = '/home/mccoyd2/Documents/AI_Hemorrhage_Detection/Reports'
report_name = 'Syngo Full%252DText Report Query-3.xlsx'
# Importing the dataset
reports = pd.read_excel(os.path.join(report_path, report_name),header = 0)
## define input parameters 
outcome = 'Hemorrhage (yes = 1, no = 0)'
impressions = 'Impression'

def numberOfNonNans(reports_column):
    count = 0
    for i in reports_column:
        if not pd.isnull(i):
            count += 1
    return count 

def define_apply_data(reports): 
    
    length_null = numberOfNonNans(reports[outcome])
    ## makes the reports to apply the model to 
    reports_apply = reports[length_null:-1]
    reports_apply.drop(reports_apply.columns[len(reports_apply.columns)-1], axis=1, inplace=True)
    reports_apply = reports_apply.reset_index(drop=True)

    
    reports_train = reports[:length_null]
    reports_train = reports_train.replace('n/a',np.NaN)
    reports_train = reports_train[np.isfinite(reports_train[outcome])]
    reports_train = reports_train.reset_index(drop=True)

    return reports_apply, reports_train

reports_apply, reports_train = define_apply_data(reports)

def report_preprocessing(reports, impressions, vocabulary, vocab_set = 1):

    ## calculate the accuracy and other metrics for bao's program 
    ## binarize the results from bao's regex program for hedging detection
    
    ##Begin preprocessing for machine learning using word to vec 
    # Cleaning the texts for word to vector
    nltk.download('stopwords')
    
    ## create a corpus of lowered, stemmed, and stop words removed
    corpus = []
    n = reports.shape[0]
    print("Creating word to vector dataframe after cleaning reports of stopwords etc.")
    for i in range(reports.shape[0]):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        j = (i + 1) / n
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
        
        sys.stdout.flush()
        sleep(0.25)
        review = re.sub('[^a-zA-Z]', ' ', reports[impressions][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
        
        if vocab_set == 0: 
            cv = CountVectorizer(max_features = 1500)
        else: 
            cv = CountVectorizer(vocabulary=vocabulary)
            
        X = cv.fit_transform(corpus).toarray()
        X = pd.DataFrame(X)
        text_features = cv.get_feature_names()
        vocabulary = cv.vocabulary_
        X.columns = text_features
        
    return X, corpus, text_features, vocabulary 

X, corpus, text_features, vocabulary = report_preprocessing(reports_train, impressions)       
        
def split_data(X, outcome):
    
    y = reports.loc[:][outcome].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


corpus, X, y, X_train, X_test, y_train, y_test = report_preprocessing(CT_reports, impressions, outcome)

def calc_metrics(y_test, prediction):
    
    cm = confusion_matrix(y_test, prediction)
    TN, FP, FN, TP = confusion_matrix(y_test, prediction).ravel()
    print(classification_report(y_test, prediction))
    Accuracy = (TP + TN)/(TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_Score = 2 * Precision * Recall / (Precision + Recall)
    
    return Accuracy, Precision, Recall, F1_Score, cm
    


def bayes_classifier(X, y):    
    
    
    # Fitting Naive Bayes to the Training set
    classifier_GNB = GaussianNB()
    classifier_GNB.fit(X_train, y_train)
    
    # Predicting the Test set results
    prediction = classifier_GNB.predict(X_test)
    
    # Making the Confusion Matrix for the Bayesian approach
    Accuracy_Bayes, Precision_Bayes, Recall_Bayes, F1_Score_Bayes, cm_Bayes = calc_metrics(y_test, prediction)
    
    return Accuracy_Bayes, Precision_Bayes, Recall_Bayes, F1_Score_Bayes, cm_Bayes
    
Accuracy_Bayes, Precision_Bayes, Recall_Bayes, F1_Score_Bayes, cm_Bayes = bayes_classifier(X, y) 

##Try even other classification models such as cART c5.0, max entroy, random forest, xgboost

# Fitting cART to the Training set

def random_forest_classifier(X, y):    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
    
    # Fitting Naive Bayes to the Training set
    classifier_RF = RandomForestClassifier(n_estimators = 10000, criterion = 'entropy', random_state = 0)
    classifier_RF.fit(X_train, y_train)
    
    # Predicting the Test set results
    prediction = classifier_RF.predict(X_test)
    
    # Making the Confusion Matrix for the Bayesian approach
    Accuracy_RF, Precision_RF, Recall_RF, F1_Score_RF, cm_RF = calc_metrics(y_test, prediction)
    
    return Accuracy_RF, Precision_RF, Recall_RF, F1_Score_RF, cm_RF, classifier_RF
    
Accuracy_RF, Precision_RF, Recall_RF, F1_Score_RF, cm_RF, classifier_RF = random_forest_classifier(X, y) 

def random_forest_feature_plot(random_forest_model, X_train): 
    ## plot the importance 
    std = np.std([tree.feature_importances_ for tree in random_forest_model.estimators_],
                 axis=0)
    
    feats = {} # a dict to hold feature_name: feature_importance
    for feature, importance, std in zip(X_train.columns, random_forest_model.feature_importances_, std):
        feats[feature] = importance, std #add the name/value pair 
    
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance',1:'STD'})
    importances = importances[importances > 0.01] ## remove zero important features
    importances = importances.dropna()
    importances = importances.sort_values(by='Gini-importance')
    
#    fig = plt.figure()
#    ax = plt.subplot(111)
    importances.plot(kind='bar', rot=45, yerr = 'STD', title = "Random forest feature importances")
    plt.savefig('RF_FeatImp.png')

    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(len(importances)):
        if importances['Gini-importance'][f] > 0.001: 
            print("%d. feature %s (%f) " % (f + 1, importances.index.values[f], importances['Gini-importance'][f]))
    
#    # Plot the feature importances of the forest
#    plt.figure()
#    plt.title("Feature importances")
#    plt.bar(range(X_train.shape[1]), importances[indices],
#           color="r", yerr=std[indices], align="center")
#    plt.xticks(range(X_train.shape[1]), indices)
#    plt.xlim([-1, X_train.shape[1]])
#    plt.show() 


# Fitting XGB - using the fit method (haven't figured out how to extract features this way)
classifier_XGB = XGBClassifier() 
classifier_XGB.fit(X_train, y_train)

y_pred = classifier_XGB.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
Accuracy_XGB = (TP + TN) / (TP + TN + FP + FN)
Precision_XGB = TP / (TP + FP)
Recall_XGB = TP / (TP + FN)
F1_Score_XGB = 2 * Precision_XGB * Recall_XGB / (Precision_XGB + Recall_XGB)

print(classifier_XGB.feature_importances_)
plt.bar(range(len(classifier_XGB.feature_importances_)), classifier_XGB.feature_importances_)
plt.show()

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(classifier_XGB, ax= ax)


### testing extraction of fscores 
# Create our DMatrix to make XGBoost more efficient

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()
    
    
## store data in a a DMatrix object for xgboost
xgdmat_train = xgb.DMatrix(X_train, y_train)
xgdmat_test = xgb.DMatrix(X_test, y_test)
## set some initial params
params = {'eta': 0.01, 'seed':0, 'subsample': 0.5, 'colsample_bytree': 0.5, 
             'objective': 'reg:linear', 'max_depth':6, 'min_child_weight':3} 

num_rounds = 50000

##train the model
mdl = xgb.train(params, xgdmat_train, num_boost_round=num_rounds)

y_pred = mdl.predict(xgdmat_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

## save the metrics from cm
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
Accuracy_XGB = (TP + TN) / (TP + TN + FP + FN)
Precision_XGB = TP / (TP + FP)
Recall_XGB = TP / (TP + FN)
F1_Score_XGB = 2 * Precision_XGB * Recall_XGB / (Precision_XGB + Recall_XGB)

## get the top features from the d matrix train method 
features = [x for x in X_train.columns if x not in ['id','loss']]
create_feature_map(features)

importance = mdl.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
df_subset = df.query('fscore > 0.01')

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')

print(df.sort_values(by='fscore', ascending=False))


## grid search xgboost for best parameters  (to get the best accuracy)

parameters_large = {'nthread':[2,3,4,5], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.01,0.02,0.05], #so called `eta` value
              'max_depth': [2,4,6,8],
              'min_child_weight': [3,5,7,9],
              'silent': [1],
              'subsample': [0.6,0.7,0.8,0.9],
              'colsample_bytree': [0.6,0.7,0.8,0.9],
              'n_estimators': [500,1000,10000], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}

parameters_small = {'nthread':[2,3], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.01,0.02], #so called `eta` value
              'max_depth': [2,4,6,8],
              'min_child_weight': [3,5,7,9],
              'silent': [1],
              'subsample': [0.6,0.7,0.8,0.9],
              'colsample_bytree': [0.6,0.7],
              'n_estimators': [100], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}

def grid_search_xg_boost(X_train, y_train, X_test, y_test, parameters):
    
    xgdmat_train = xgb.DMatrix(X_train, y_train)
    xgdmat_test = xgb.DMatrix(X_test, y_test)
    
    xgb_model = xgb.XGBClassifier()
    
    clf = GridSearchCV(xgb_model, parameters_small, n_jobs=1000, 
                       cv=StratifiedKFold(y_train, n_folds=10, shuffle=True), 
                       scoring='roc_auc',
                       verbose=10, refit=True)
    
    clf.fit(X_train, y_train)
    
    pickle.dump(clf, open("pima.pickle.dat", "wb")) ## save the model
    
    # load model from file
    loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
    
    ## exract the best scores from the grid search
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score)
    
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    
    predictions = clf.predict(X_test)

    Accuracy_XGB, Precision_XGB, Recall_XGB, F1_Score_XGB, cm_XGB = calc_metrics(y_test, predictions)
    
    num_rounds = 1000
    
    mdl_grid_best = xgb.train(best_parameters, xgdmat_train, num_boost_round=num_rounds)
    y_pred_best_grid = mdl_grid_best.predict(xgdmat_test)
    y_pred_best_grid = np.where(y_pred_best_grid > 0.5, 1, 0)
    
    Accuracy_bestXG, Precision_bestXG, Recall_bestXG, F1_Score_bestXG, cm_bestXG = calc_metrics(y_test, y_pred_best_grid)

    ## get the top features from the d matrix train method 
    features_best_grid = [x for x in X_train.columns if x not in ['id','loss']]
    create_feature_map_best_grid(features_best_grid)
    
    importance_bg = mdl_grid_best.get_fscore(fmap='xgb_best_grid.fmap')
    importance_bg = sorted(importance_bg.items(), key=operator.itemgetter(1))
    
    df_bg = pd.DataFrame(importance_bg, columns=['feature', 'fscore'])
    df_bg['fscore'] = df_bg['fscore'] / df_bg['fscore'].sum()
    df_bg_subset = df_bg.query('fscore > 0.01')
    
    plt.figure()
    df_bg.plot()
    df_bg.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.gcf().savefig('feature_importance_xgb.png')
    
    print(df.sort_values(by='fscore', ascending=False))
    
    ## plot same or subset of f > 0.01
    plt.figure()
    df_bg_subset.plot()
    df_bg_subset.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
    plt.title('XGBoost Feature Importance Subset F < 0.01')
    plt.xlabel('relative importance')
    plt.gcf().savefig('feature_importance_xgb.png')
    
    print(df_bg_subset.sort_values(by='fscore', ascending=False))

    
##get features from the best xgboost params 

## create function to save the best grid feature fscores to an fmap for extraction
def create_feature_map_best_grid(features):
    outfile = open('xgb_best_grid.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()
    

def create_metrics_table(): 
    Accuracies =  Accuracy_Bayes, Accuracy_RF, Accuracy_XGB, Accuracy_bestXG
    Precisions = Precision_Bayes, Precision_RF, Precision_XGB, Precision_bestXG
    Recalls = Recall_Bayes, Recall_RF, Recall_XGB, Recall_bestXG
    F1_Scores = F1_Score_Bayes, F1_Score_RF, F1_Score_XGB, F1_Score_bestXG
    
    Metrics = Accuracies, Precisions, Recalls, F1_Scores
    Methods = ["Regex", "Bayesian","Random Forest", "Grid Search XG Boost"]
    Metrics = pd.DataFrame(list(Metrics),columns= Methods, index = ['Accuracy', 'Precision', 'Recall', 'F1 score'])
    
    ## save the results
    Metrics.to_csv("/home/mccoyd2/Dropbox/Uncertainty Project/Hedging_ML_Metrics.csv", sep=',')

def apply_model(reports_apply, impressions, vocabulary):
    
    X_apply, corpus_apply, text_features_apply, vocabulary_apply = report_preprocessing(reports_apply, impressions, vocabulary, vocab_set = 1 )       
    xgdmat_appy = xgb.DMatrix(X_apply)
    all_data_pred_best_grid = mdl_grid_best.predict(xgdmat_appy)
    
    all_data_pred_best_grid = pd.DataFrame(all_data_pred_best_grid)
    
    all_data_pred_best_grid_cat = np.where(all_data_pred_best_grid > 0.90, 'high confidence hemorrhage', 
         (np.where(all_data_pred_best_grid < 0.20, 'high confidence no hemorrhage', 'medium confidence hemorrhage')))

    all_data_pred_best_grid_cat = pd.DataFrame(all_data_pred_best_grid_cat)

    apply_predictions_concat = pd.concat([all_data_pred_best_grid, all_data_pred_best_grid_cat], axis=1)
    
    apply_predictions_concat_reports = pd.concat([reports_apply, apply_predictions_concat], axis=1)
    list(apply_predictions_concat_reports.columns.values)
    apply_predictions_concat_reports.columns.values[17] = 'Prediction Probability'
    apply_predictions_concat_reports.columns.values[18] = 'Confidence Category'
    apply_predictions_concat_reports.to_csv("/home/mccoyd2/Documents/AI_Hemorrhage_Detection/Reports/Predictions/Hemorrhage_Reports_Batch_1_Predictions.csv", sep=',')

    return apply_predictions_concat_reports

def get_group_list_accessions(model_applied_reports):
    hemorrhage_case_accessions = [] 
    hemorrhage_control_accessions = [] 
    hemorrhage_review_accessions = [] 
    for index in range(model_applied_reports.shape[0]):
        prediction = model_applied_reports['Confidence Category'].iloc[index]
        if prediction == 'high confidence no hemorrhage' :
            hemorrhage_control_accessions.append(model_applied_reports['Acn'].iloc[index])
        elif prediction == 'high confidence hemorrhage':
            hemorrhage_case_accessions.append(model_applied_reports['Acn'].iloc[index])
        else: 
            hemorrhage_review_accessions.append(model_applied_reports['Acn'].iloc[index])
            
    return hemorrhage_control_accessions, hemorrhage_case_accessions, hemorrhage_case_accessions

hemorrhage_case_accessions = map(str, hemorrhage_case_accessions)
hemorrhage_review_accessions = map(str, hemorrhage_review_accessions)
hemorrhage_control_accessions = map(str, hemorrhage_control_accessions)

def save_accession_sql_format(accessions,name):
    f = open(name, "w")
    
    out = ""
    for i, num in enumerate(accessions):
    	if i%1000 != 0:
    		out += "\'"+num+"\', "
    	else: 
    		out = out[:-2]
    		out += "\n"
    		f.write(out)
    		out = ""
    
    
    out = out[:-2]
    out += "\n"
    f.write(out)
    
    f.close()

save_accession_sql_format(hemorrhage_case_accessions,'hemorrhage_case_accessions_batch1.txt')
save_accession_sql_format(hemorrhage_review_accessions,'hemorrhage_review_accessions_batch1.txt')
save_accession_sql_format(hemorrhage_control_accessions,'hemorrhage_control_accessions_batch1.txt')
