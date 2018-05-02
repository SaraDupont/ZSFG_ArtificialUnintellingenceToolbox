"""
Created on Wed Oct 11 11:13:39 2017

@author: davidbmccoy
"""
# Natural Language Processing to Detect Hedging in Radiology Reports 

# Importing the libraries
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import shutil
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
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
import argparse


def get_parser_classify():
    # classification parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-data-path",
                        help="Path to the data folder",
                        type=str,
                        dest="data_path")
    parser.add_argument("-data-name",
                        help="Name of the reports file",
                        type=str,
                        dest="data_name")
    parser.add_argument("-outcome",
                        help="Name of column in sheet for outcome",
                        type=str,
                        dest="outcome")
    parser.add_argument("-impressions",
                        help="Name of column for report impressions from radiologist",
                        type=str,
                        dest="impressions",
                        default="Impression")
    parser.add_argument("-exclude-label",
                        help="Label that should not be used for report classification",
                        type=int,
                        dest="exclude_label",
                        default=np.nan)
    parser.add_argument("-apply-label",
                        help="Label for the non reviewed data (to be used as apply data)",
                        type=int,
                        dest="apply_label",
                        default=np.nan)
    parser.add_argument("-output-path",
                        help="Path for saving models and figures",
                        type=str,
                        dest="output_path",
                        default=".")
    parser.add_argument("-confidence-thr-case",
                        help="Confidence threshold used to define a case",
                        type=float,
                        dest="case_thr",
                        default=0.9)
    parser.add_argument("-confidence-thr-control",
                        help="Confidence threshold used to define a control",
                        type=float,
                        dest="control_thr",
                        default=0.2)
    parser.add_argument("-number-word-features",
                        help="Number of words to extract for word to vector",
                        type=int,
                        dest="number_word_features",
                        default=1500)
    parser.add_argument("-v",
                        help="verbose param. set to 2 to save additional graphs",
                        type=int,
                        dest="verbose",
                        default=1)
    parser.add_argument("-apply-all",
                        help="apply a model to a full dataset, no training",
                        type=bool,
                        dest="apply_all",
                        default=False)
    parser.add_argument("-no-apply",
                        help="Don't apply the model to the testing set",
                        type=bool,
                        dest="no_apply",
                        default=False)
    parser.add_argument("-skip-preprocessing",
                        help="skip preprocessing and text cleaning and use saved sparse matrix",
                        type=bool,
                        dest="skip_preprocessing",
                        default=False)
    parser.add_argument("-max-ngrams",
                        help="maximum number of ngram (adjacent words) in preprocessing",
                        type=to_list,
                        dest="max_ngrams",
                        default=[1])
    return parser

def to_list(l):
    str_l = l.split(',')
    int_l = [int(e) for e in str_l]
    return int_l

class Model_results:
    def __init__(self, method_name, accuracy, precision, recall, f1_score, cm, label_pred=None, label_true=None):
        self.method_name = method_name
        #
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.cm = cm
        #
        self.label_pred = label_pred
        self.label_true = label_true

class Radiology_Report_NLP:

    def __init__(self, param):
        self.param = param
        self.i_ngrams=0
        self.confidence_labels = ('high confidence negative', 'medium confidence positive', 'high confidence positive')
        self.parameters_large = {'nthread': [2, 3, 4, 5],  # when use hyperthread, xgboost may become slower
                                 'objective': ['binary:logistic'],
                                 'learning_rate': [0.01, 0.02, 0.05],  # so called `eta` value
                                 'max_depth': [2, 4, 6, 8],
                                 'min_child_weight': [3, 5, 7, 9],
                                 'silent': [1],
                                 'subsample': [0.6, 0.7, 0.8, 0.9],
                                 'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                                 'n_estimators': [500, 1000, 10000],
                                 # number of trees, change it to 1000 for better results
                                 'missing': [-999],
                                 'seed': [1337]}

        self.parameters_small = {'nthread': [2, 3],  # when use hyperthread, xgboost may become slower
                                 'objective': ['binary:logistic'],
                                 'learning_rate': [0.01, 0.02],  # so called `eta` value
                                 'max_depth': [2, 4, 6, 8],
                                 'min_child_weight': [3, 5, 7, 9],
                                 'silent': [1],
                                 'subsample': [0.6, 0.7, 0.8, 0.9],
                                 'colsample_bytree': [0.6, 0.7],
                                 'n_estimators': [1000],  # number of trees, change it to 1000 for better results
                                 'missing': [-999],
                                 'seed': [1337]}

        self.parameters_x_small = {'nthread': [2, 3],  # when use hyperthread, xgboost may become slower
                                   'objective': ['binary:logistic'],
                                   'learning_rate': [0.01],  # so called `eta` value
                                   'max_depth': [2, 4],
                                   'min_child_weight': [3],
                                   'silent': [1],
                                   'subsample': [0.6],
                                   'colsample_bytree': [0.6, 0.7],
                                   'n_estimators': [100],  # number of trees, change it to 1000 for better results
                                   'missing': [-999],
                                   'seed': [1337]}

        self.parameters_best = {'colsample_bytree': 0.6,
                                'silent': 1,
                                'missing': -999,
                                'learning_rate': 0.01,
                                'nthread': 2,
                                'min_child_weight': 3,
                                'n_estimators': 1000,
                                'subsample': 0.9,
                                'seed': 1337,
                                'objective': 'binary:logistic',
                                'max_depth': 2}

        if '.xls' in self.param.data_name:
            self.reports = pd.read_excel(os.path.join(self.param.data_path, self.param.data_name), header=0)
        elif '.csv' in self.param.data_name:
            self.reports = pd.read_csv(os.path.join(self.param.data_path, self.param.data_name))
        else:
            print 'ERROR: input file should be a .xls or .csv file.'
            self.reports = None

        self.folder_models = 'Models'

        self.dic_model_names = {
            'Bayesian': 'bayes_model.sav',
            'SGD': 'SGD_model.sav',
            'Logistic': 'logistic_model.sav',
            'SGD_grid': 'SGD_grid.sav',
            'RandomForest': 'random_forest_model.sav',
            'Gridsearch XGB': 'pisma.pickle.dat',
            'Best XGB': 'best_xgb.pickle.dat'}

    def preprocessing_full(self):
        if not os.path.isdir(os.path.join(self.param.output_path, self.folder_models)):
            os.mkdir(os.path.join(self.param.output_path, self.folder_models))
        # define data
        self.define_apply_data()
        folder_word2vec = 'Word2Vec'
        if not os.path.isdir(os.path.join(self.param.output_path, folder_word2vec)):
            os.mkdir(os.path.join(self.param.output_path, folder_word2vec))
        # preprocess train and apply datasets
        parent_folder = '/'.join(self.param.output_path.split('/')[:-1])
        path_corpus = parent_folder if len(self.param.max_ngrams) > 1 else self.param.output_path

        if not self.param.apply_all:
            fname_csv_train = 'Word_to_vec_X_Sparse_Matrix_train.csv'
            fname_csv_apply = 'Word_to_vec_X_Sparse_Matrix_apply.csv'
            if not self.param.skip_preprocessing:

                corpus_train = np.load(os.path.join(path_corpus, 'corpus_train.npy')) if os.path.isfile(os.path.join(path_corpus, 'corpus_train.npy')) else None
                corpus_apply = np.load(os.path.join(path_corpus, 'corpus_apply.npy')) if os.path.isfile(os.path.join(path_corpus, 'corpus_apply.npy')) else None
                self.X_train, self.text_feature_train, corpus_train_new = self.report_preprocessing(self.reports_train, vocab_set=0, corpus=corpus_train)
                self.X_apply, self.text_feature_apply, corpus_apply_new = self.report_preprocessing(self.reports_apply, vocab_set=1, corpus=corpus_apply)

                if corpus_train is None:
                    np.save(os.path.join(path_corpus, 'corpus_train.npy'), corpus_train_new)
                if corpus_apply is None:
                    np.save(os.path.join(path_corpus, 'corpus_apply.npy'), corpus_apply_new)

                self.X_train.to_csv(os.path.join(self.param.output_path, folder_word2vec, fname_csv_train))
                self.X_apply.to_csv(os.path.join(self.param.output_path, folder_word2vec, fname_csv_apply))
            else:
                self.X_train = pd.read_csv(os.path.join(self.param.output_path, folder_word2vec, fname_csv_train))
                self.X_apply = pd.read_csv(os.path.join(self.param.output_path, folder_word2vec, fname_csv_apply))

        #            self.X_apply, self.text_feature_apply = self.report_preprocessing(self.reports_apply, vocab_set=1)

        if self.param.apply_all:
            fname_csv_full = 'Word_to_vec_X_Sparse_Matrix_full_data_apply.csv'
            if not self.param.skip_preprocessing:
                self.vocabulary = pickle.load(open(self.param.output_path + "/vocabulary.pickle", "rb"))
                #
                corpus_apply_all = np.load(os.path.join(path_corpus, 'corpus_apply_all.npy')) if os.path.isfile(os.path.join(path_corpus, 'corpus_apply_all.npy')) else None
                self.X_apply, self.text_feature_apply, corpus_apply_all_new = self.report_preprocessing(self.reports_apply, vocab_set=1, corpus=corpus_apply_all)
                self.X_apply.to_csv(os.path.join(self.param.output_path, folder_word2vec, fname_csv_full))
                if corpus_apply_all is None:
                    np.save(os.path.join(path_corpus, 'corpus_apply_all.npy'), corpus_apply_all_new)

            else:
                self.X_apply = pd.read_csv(os.path.join(self.param.output_path, folder_word2vec, fname_csv_full))
        #
        if 'Unnamed: 0' in self.X_apply.columns:
            self.X_apply = self.X_apply.drop('Unnamed: 0', axis=1)

        self.split_data()

    def define_apply_data(self):
        self.reports = self.drop_columns(self.reports)
        # remove NAN impressions
        self.reports = self.reports.dropna(axis=0, how='any')

        if not self.param.apply_all:
            if not np.isnan(self.param.exclude_label):
                self.reports = self.reports.drop(self.reports[self.reports[self.param.outcome] == int(self.param.exclude_label)].index)

            if np.isnan(self.param.apply_label):
                self.reports_apply = self.reports[self.reports[self.param.outcome].isnull()]
            else:
                self.reports_apply = self.reports[self.reports[self.param.outcome] == self.param.apply_label]

            self.reports_apply = self.reports_apply.reset_index(drop=True)
            self.reports_apply = self.reports_apply[:-1]
            # self.reports_apply = self.reports_apply.drop(self.param.outcome, 1)
            # self.reports_apply = self.drop_columns(self.reports_apply)

            if np.isnan(self.param.apply_label):
                self.reports_train = self.reports[self.reports[self.param.outcome].notnull()]
            else:
                self.reports_train = self.reports[self.reports[self.param.outcome] != self.param.apply_label]
            #
            # self.reports_train = self.drop_columns(self.reports_train)
            #
            self.reports_train = self.reports_train.drop_duplicates()

            self.reports_train = self.reports_train.reset_index(drop=True)

        if self.param.apply_all:
            self.reports_apply = self.reports

    def drop_columns(self, reports):
        for col in reports.columns:
            if col not in [self.param.outcome, self.param.impressions]:
                reports = reports.drop(col, 1)
        return reports

    def report_preprocessing(self, reports, vocab_set, corpus=None):

        nltk.download('stopwords')

        if corpus is None:
            ## create a corpus of lowered, stemmed, and stop words removed
            corpus = []
            n = reports.shape[0]
            print("Creating word to vector dataframe after cleaning reports of stopwords etc.")
            for i in range(reports.shape[0]):
                sys.stdout.write('\r')
                # the exact output you're looking for:
                j = (i + 1) / float(n)
                sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))

                sys.stdout.flush()
                sleep(0.25)
                # review = re.sub('(?<=Dr.)(.{20}(?:\s|.))', '',reports[self.param.impressions][i])
                review = re.sub(r'[Dd]iscussed.*\d', '', reports[self.param.impressions].iloc[i], flags=re.DOTALL)
                review = re.sub('[^a-zA-Z]', ' ', review)
                review = review.lower()
                review = review.split()
                ps = PorterStemmer()
                review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
                review = ' '.join(review)
                corpus.append(review)

        if vocab_set == 0:
            cv = CountVectorizer(max_features=self.param.number_word_features, ngram_range=(1, self.param.max_ngrams[self.i_ngrams]))
            X = cv.fit_transform(corpus).toarray()
            self.vocabulary = cv.vocabulary_
            # save to pickle
            pickle_out = open(self.param.output_path + "/vocabulary.pickle", "wb")
            pickle.dump(self.vocabulary, pickle_out)
            pickle_out.close()

        else:
            cv = CountVectorizer(vocabulary=self.vocabulary, ngram_range=(1, self.param.max_ngrams[self.i_ngrams]))
            X = cv.fit_transform(corpus).toarray()

        text_features = cv.get_feature_names()

        X = pd.DataFrame(X)
        X.columns = text_features

        return X, text_features, corpus

    def split_data(self):
        if not self.param.apply_all:
            self.y_train = self.reports_train[self.param.outcome]
            self.X_train, self.X_valid, self.y_train, self.y_valid, self.reports_train, self.reports_valid = train_test_split(self.X_train, self.y_train, self.reports_train, test_size=0.30, random_state=0)

    def run_models(self):
        bayes_res = self.bayes_classifier()
        SGD_res, Logistic_res = self.SGD_Logistic_classifier()
        SGD_grid_res = self.SGD_Grid_classifier()
        rf_res = self.random_forest_classifier()
        # xgb_res, best_xgb_res, self.model = self.grid_search_xg_boost(self.parameters_x_small)
        xgb_res, best_xgb_res, self.model = self.grid_search_xg_boost(self.parameters_small)
        # xgb_res, best_xgb_res, self.model = self.grid_search_xg_boost(self.parameters_large)

        # list_res = [bayes_res, SGD_res, Logistic_res, rf_res]
        list_res = [bayes_res, SGD_res, Logistic_res, SGD_grid_res, rf_res, xgb_res, best_xgb_res]
        for res in list_res:
            pickle.dump(res, open(os.path.join(self.param.output_path, self.folder_models, res.method_name+'.pkl'), 'wb'))
        pickle.dump((self.X_valid, self.y_valid), open(os.path.join(self.param.output_path, self.folder_models, 'X_y_valid.pkl'), 'wb'))
        pickle.dump(self.reports_valid, open(os.path.join(self.param.output_path, self.folder_models, 'reports_valid.pkl'), 'wb'))

        self.create_metrics_table(list_res)

    def bayes_classifier(self):
        # Fitting Naive Bayes to the Training set
        model_name = 'Bayesian'
        classifier_GNB = GaussianNB()
        classifier_GNB.fit(self.X_train, self.y_train)
        bayes_prediction = classifier_GNB.predict(self.X_valid)

        Accuracy_bayes, Precision_bayes, Recall_bayes, F1_Score_bayes, cm_bayes = calc_metrics(self.y_valid,
                                                                                               bayes_prediction)

        bayes_res = Model_results(model_name, Accuracy_bayes, Precision_bayes, Recall_bayes, F1_Score_bayes, cm_bayes, label_pred=bayes_prediction, label_true=self.y_valid)

        pickle.dump(classifier_GNB, open(os.path.join(self.param.output_path, self.folder_models, self.dic_model_names[model_name]), 'wb'))

        return bayes_res

    def SGD_Logistic_classifier(self):
        # Fitting Naive Bayes to the Training set
        numFolds = 10
        kf = KFold(self.X_train.shape[0], numFolds, shuffle=True)

        # These are "Class objects". For each Class, find the AUC through
        # 10 fold cross validation.
        Models = [LogisticRegression, SGDClassifier]
        params = [{}, {"loss": "log", "penalty": "l2", 'max_iter': 1000}]
        model_names = ['Logistic', 'SGD']

        accuracy, precision, recall, f1_score, cm, fitted_models = [0] * len(Models), [0] * len(Models), [0] * len(Models), [0] * len( Models), [0] * len(Models), [None]*len(Models)

        for i, (param, Model, model_name) in enumerate(zip(params, Models, model_names)):
            total = 0
            for train_indices, test_indices in kf:
                train_X = self.X_train.iloc[train_indices]
                train_Y = self.y_train.iloc[train_indices]
                test_X = self.X_train.iloc[test_indices]
                test_Y = self.y_train.iloc[test_indices]

                reg = Model(**param)
                reg.fit(train_X, train_Y)
                predictions = reg.predict(test_X)

                #
                fitted_models[i] = reg
                model_accuracy, model_precision, model_recall, model_f1, model_cm = calc_metrics(test_Y, predictions)

                accuracy[i] += model_accuracy
                precision[i] += model_precision
                recall[i] += model_recall
                f1_score[i] += model_f1
                cm[i] += model_cm

                total += accuracy_score(test_Y, predictions)

            pickle.dump(reg, open(os.path.join(self.param.output_path, self.folder_models,  self.dic_model_names[model_name]), 'wb'))
            total_accuracy = total / numFolds
            print "Accuracy score of {0}: {1}".format(Model.__name__, total_accuracy)

        # average
        accuracy = [val / numFolds for val in accuracy]
        precision = [val / numFolds for val in precision]
        recall = [val / numFolds for val in recall]
        f1_score = [val / numFolds for val in f1_score]
        cm = [val / numFolds for val in cm]

        ### BE CAREFUL TO KEEP THE SAME ORDER HERE
        model_name_log = 'Logistic'
        model_name_sgd = 'SGD'
        i_log, i_sgd = 0, 1
        pred_log = fitted_models[i_log].predict(self.X_valid)
        pred_sgd = fitted_models[i_sgd].predict(self.X_valid)
        Logistic_res = Model_results(model_names[i_log], accuracy[i_log], precision[i_log], recall[i_log], f1_score[i_log], cm[i_log], label_true = self.y_valid, label_pred=pred_log)
        SGD_res = Model_results(model_names[i_sgd], accuracy[i_sgd], precision[i_sgd], recall[i_sgd], f1_score[i_sgd], cm[i_sgd], label_true = self.y_valid, label_pred=pred_sgd)

        return SGD_res, Logistic_res

    def SGD_Grid_classifier(self):
        model_name = 'SGD_grid'
        parameters = {
            'loss': ['log'],
            'penalty': ['elasticnet'],
            'alpha': [10 ** x for x in range(-6, 1)],
            'l1_ratio': [0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1],
        }

        clf = SGDClassifier(loss="log", random_state=0, class_weight='balanced')

        clf_grid = GridSearchCV(estimator=clf, param_grid=parameters, n_jobs=1000,
                                cv=StratifiedKFold(self.y_train, n_folds=10, shuffle=True),
                                scoring='roc_auc',
                                verbose=10)
        #
        clf_grid.fit(self.X_train, self.y_train)
        print("Best score: %0.3f" % clf_grid.best_score_)
        print("Best parameters set:")
        best_parameters = clf_grid.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        SGD_grid_pred = clf_grid.predict(self.X_valid)
        pickle.dump(clf_grid, open(os.path.join(self.param.output_path, self.folder_models, self.dic_model_names[model_name]), "wb"))  ## save the model

        Accuracy_SGD_grid, Precision_SGD_grid, Recall_SGD_grid, F1_Score_XGD_grid, cm_XGD_grid = calc_metrics(
            self.y_valid, SGD_grid_pred)

        SGD_grid_res = Model_results(model_name, Accuracy_SGD_grid, Precision_SGD_grid, Recall_SGD_grid, F1_Score_XGD_grid, cm_XGD_grid, label_true=self.y_valid, label_pred=SGD_grid_pred)

        return SGD_grid_res

    def random_forest_classifier(self):
        model_name = 'RandomForest'

        classifier_RF = RandomForestClassifier(n_estimators=10000, criterion='entropy', random_state=0)
        classifier_RF.fit(self.X_train, self.y_train)

        if self.param.verbose == 2:
            self.random_forest_feature_plot(classifier_RF)

        # Predicting the Test set results
        RF_prediction = classifier_RF.predict(self.X_valid)

        Accuracy_rf, Precision_rf, Recall_rf, F1_Score_rf, cm_rf = calc_metrics(self.y_valid, RF_prediction)

        rf_res = Model_results(model_name, Accuracy_rf, Precision_rf, Recall_rf, F1_Score_rf, cm_rf, label_true=self.y_valid, label_pred=RF_prediction)

        pickle.dump(classifier_RF, open(os.path.join(self.param.output_path, self.folder_models, self.dic_model_names[model_name]), 'wb'))

        return rf_res

    def random_forest_feature_plot(self, rf_classifier):
        ## plot the importance
        std = np.std([tree.feature_importances_ for tree in rf_classifier.estimators_], axis=0)

        feats = {}  # a dict to hold feature_name: feature_importance
        for feature, importance, std in zip(self.X_train.columns, rf_classifier.feature_importances_, std):
            feats[feature] = importance, std  # add the name/value pair

        importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance', 1: 'STD'})
        importances = importances[importances > 0.01]  ## remove zero important features
        importances = importances.dropna()
        importances = importances.sort_values(by='Gini-importance')

        #    fig = plt.figure()
        #    ax = plt.subplot(111)
        importances.plot(kind='bar', rot=45, yerr='STD', title="Random forest feature importances")
        plt.savefig('RF_FeatImp.png')

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(len(importances)):
            if importances['Gini-importance'][f] > 0.001:
                print("%d. feature %s (%f) " % (f + 1, importances.index.values[f], importances['Gini-importance'][f]))

    def create_feature_map(self, features):
        outfile = open('xgb.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1

        outfile.close()

    def grid_search_xg_boost(self, parameters):
        model_name_grid = 'Gridsearch XGB'
        model_name_best = 'Best XGB'

        xgdmat_train = xgb.DMatrix(self.X_train, self.y_train)
        xgdmat_test = xgb.DMatrix(self.X_valid, self.y_valid)

        xgb_model = xgb.XGBClassifier()

        clf = GridSearchCV(xgb_model, parameters, n_jobs=1000,
                           cv=StratifiedKFold(self.y_train, n_folds=10, shuffle=True),
                           scoring='roc_auc',
                           verbose=10, refit=True)

        clf.fit(self.X_train, self.y_train)
        pickle.dump(clf, open(os.path.join(self.param.output_path, self.folder_models, self.dic_model_names[model_name_grid]), "wb"))  ## save the model

        # load model from file
        # loaded_model = pickle.load(open("pima.pickle.dat", "rb"))

        ## exract the best scores from the grid search
        best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
        print('Raw AUC score:', score)

        for param_name in sorted(best_parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))

        predictions = clf.predict(self.X_valid)

        #

        #    ## store data in a a DMatrix object for xgboost
        #    xgdmat_train = xgb.DMatrix(X_train, y_train)
        #    xgdmat_test = xgb.DMatrix(X_test, y_test)
        #    ## set some initial params
        #    params = {'eta': 0.01, 'seed':0, 'subsample': 0.5, 'colsample_bytree': 0.5,
        #                 'objective': 'reg:linear', 'max_depth':6, 'min_child_weight':3}
        #
        #    num_rounds = 50000
        #
        #    ##train the model
        #    mdl = xgb.train(params, xgdmat_train, num_boost_round=num_rounds)
        #
        #    y_pred = mdl.predict(xgdmat_test)
        #    y_pred = np.where(y_pred > 0.5, 1, 0)
        #    cm = confusion_matrix(y_test, y_pred)
        #    print(classification_report(y_test, y_pred))
        #
        #    ## save the metrics from cm
        #    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
        #    Accuracy_XGB = (TP + TN) / (TP + TN + FP + FN)
        #    Precision_XGB = TP / (TP + FP)
        #    Recall_XGB = TP / (TP + FN)
        #    F1_Score_XGB = 2 * Precision_XGB * Recall_XGB / (Precision_XGB + Recall_XGB)
        #
        #    ## get the top features from the d matrix train method
        #    features = [x for x in X_train.columns if x not in ['id','loss']]
        #    self.create_feature_map(features)
        #
        #    importance = mdl.get_fscore(fmap='xgb.fmap')
        #    importance = sorted(importance.items(), key=operator.itemgetter(1))
        #
        #    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        #    df['fscore'] = df['fscore'] / df['fscore'].sum()
        #    df_subset = df.query('fscore > 0.01')
        #
        #    plt.figure()
        #    df.plot()
        #    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
        #    plt.title('XGBoost Feature Importance')
        #    plt.xlabel('relative importance')
        #    plt.gcf().savefig('feature_importance_xgb.png')
        #
        #    print(df.sort_values(by='fscore', ascending=False))

        ## grid search xgboost for best parameters  (to get the best accuracy)
        Accuracy_XGB, Precision_XGB, Recall_XGB, F1_Score_XGB, cm_XGB = calc_metrics(self.y_valid, predictions)
        xgb_res = Model_results(model_name_grid, Accuracy_XGB, Precision_XGB, Recall_XGB, F1_Score_XGB, cm_XGB, label_true=self.y_valid, label_pred=predictions)

        num_rounds = 1000

        mdl_grid_best = xgb.train(best_parameters, xgdmat_train, num_boost_round=num_rounds)
        y_pred_best_grid = mdl_grid_best.predict(xgdmat_test)
        y_pred_best_grid = np.where(y_pred_best_grid > 0.5, 1, 0)

        Accuracy_bestXG, Precision_bestXG, Recall_bestXG, F1_Score_bestXG, cm_bestXG = calc_metrics(self.y_valid,
                                                                                                    y_pred_best_grid)
        best_xgb_res = Model_results(model_name_best, Accuracy_bestXG, Precision_bestXG, Recall_bestXG, F1_Score_bestXG, cm_bestXG, label_true=self.y_valid, label_pred=y_pred_best_grid)

        pickle.dump(mdl_grid_best, open(os.path.join(self.param.output_path, self.folder_models, self.dic_model_names[model_name_best]), "wb"))

        if self.param.verbose == 2:
            ## get the top features from the d matrix train method
            features_best_grid = [x for x in self.X_train.columns if x not in ['id', 'loss']]
            features_best_grid = features_best_grid[1:]
            self.create_feature_map_best_grid(features_best_grid)

            importance_bg = mdl_grid_best.get_fscore(fmap=self.param.output_path + '/xgb_best_grid.fmap')
            importance_bg = sorted(importance_bg.items(), key=operator.itemgetter(1))

            df_bg = pd.DataFrame(importance_bg, columns=['feature', 'fscore'])
            df_bg['fscore'] = df_bg['fscore'] / df_bg['fscore'].sum()
            df_bg_subset = df_bg.query('fscore > 0.01')

            plt.figure()
            df_bg.plot()
            df_bg.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
            plt.title('XGBoost Feature Importance')
            plt.xlabel('relative importance')
            plt.gcf().savefig(self.param.output_path + '/feature_importance_xgb.png')

            print(df_bg.sort_values(by='fscore', ascending=False))

            ## plot same or subset of f > 0.01
            plt.figure()
            df_bg_subset.plot()
            df_bg_subset.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
            plt.title('XGBoost Feature Importance Subset F < 0.01')
            plt.xlabel('relative importance')
            plt.gcf().savefig(self.param.output_path + '/feature_importance_xgb_subset.png')

            print(df_bg_subset.sort_values(by='fscore', ascending=False))

        return xgb_res, best_xgb_res, mdl_grid_best

    ##get features from the best xgboost params

    ## create function to save the best grid feature fscores to an fmap for extraction
    def create_feature_map_best_grid(self, features):
        outfile = open(self.param.output_path + '/xgb_best_grid.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1

        outfile.close()

    def create_metrics_table(self, list_res):
        accuracies = [res.accuracy for res in list_res]
        precisions = [res.precision for res in list_res]
        recalls = [res.recall for res in list_res]
        f1_scores = [res.f1_score for res in list_res]
        methods = [res.method_name for res in list_res]

        metrics = [accuracies, precisions, recalls, f1_scores]

        df_metrics = pd.DataFrame(metrics, columns=methods, index=['Accuracy', 'Precision', 'Recall', 'F1 score'])

        ## save the results
        df_metrics.to_csv(self.param.output_path + '/Metrics_Table.csv', sep=',')

    def apply_model(self, model_name):
        #        xgb_status=False

        fname_model = self.dic_model_names[model_name]
        xgb_status = True if model_name == "Best XGB" else False
        #        if model == "Bayesian" :
        #            fname_model = 'bayes_model.sav'
        #        if model == "RandomForest" :
        #            fname_model = 'random_forest_model.sav'
        #        if model == "Gridsearch XGB" :
        #            fname_model = "pima.pickle.dat"
        #        if model == "SGD" :
        #            fname_model = "SGD_model.sav"
        #        if model == "Best XGB":
        #            fname_model = "best_xgb.pickle.dat"
        #            xgb_status=True

        model = pickle.load(open(os.path.join(self.param.output_path, self.folder_models, fname_model), "rb"))

        if not xgb_status:
            x_apply = self.X_apply
            pred = model.predict_proba(x_apply)
            df_pred = pd.DataFrame(pred)
            df_pred = df_pred[1]
        else:
            x_apply = xgb.DMatrix(self.X_apply)
            pred = model.predict(x_apply)
            df_pred = pd.DataFrame(pred)

        #### check if other models than xgb give percentages of confidence
        df_pred_cat = np.where(df_pred > self.param.case_thr, self.confidence_labels[2],
                               (np.where(df_pred < self.param.control_thr, self.confidence_labels[0],
                                         self.confidence_labels[1])))

        df_case_control = np.where(df_pred > self.param.case_thr, 1,
                                   (np.where(df_pred < self.param.control_thr, 0, 2)))

        df_pred_cat = pd.DataFrame(df_pred_cat)
        df_case_control = pd.DataFrame(df_case_control)

        apply_predictions_concat = pd.concat([df_pred, df_pred_cat, df_case_control], axis=1)

        apply_predictions_concat_reports = pd.concat([self.reports_apply, apply_predictions_concat], axis=1)
        ##### check column names while debugging
        #        list(apply_predictions_concat_reports.columns)
        # apply_predictions_concat_reports.columns.values[-3] = 'Prediction Probability'
        # apply_predictions_concat_reports.columns.values[-2] = 'Confidence Category'
        # apply_predictions_concat_reports.columns.values[-1] = 'Osteomyelitis'

        if not self.param.apply_all:
            apply_predictions_concat_reports.to_csv(self.param.output_path + "/Predictions.csv", sep=',')

        if self.param.apply_all:
            apply_predictions_concat_reports.to_csv(self.param.output_path + "/Predictions_Full_Apply.csv", sep=',')

        return apply_predictions_concat_reports

    def get_group_list_accessions(self, model_applied_reports):

        case_accessions = []
        control_accessions = []
        review_accessions = []

        case_accessions_train = []
        control_accessions_train = []
        review_accessions_train = []

        count_cases_apply = 0
        count_controls_apply = 0

        count_cases_train = 0
        count_controls_train = 0

        if not self.param.apply_all:

            for index in range(model_applied_reports.shape[0]):
                prediction = model_applied_reports['Confidence Category'].iloc[index]
                if prediction == self.confidence_labels[0]:
                    control_accessions.append(model_applied_reports['Accession1'].iloc[index])
                    count_controls_apply += 1

                elif prediction == self.confidence_labels[2]:
                    case_accessions.append(model_applied_reports['Accession1'].iloc[index])
                    count_cases_apply += 1
                else:
                    review_accessions.append(model_applied_reports['Accession1'].iloc[index])

            for index in range(self.reports_train.shape[0]):
                label = self.reports_train[self.param.outcome].iloc[index]
                if label == 0:
                    control_accessions_train.append(self.reports_train['Accession1'].iloc[index])
                    count_controls_train += 1
                elif label == 1:
                    case_accessions_train.append(self.reports_train['Accession1'].iloc[index])
                    count_cases_train += 1
                else:
                    review_accessions_train.append(self.reports_train['Accession1'].iloc[index])

            control_accessions_train, case_accessions_train, review_accessions_train = map(str,
                                                                                           control_accessions_train), map(
                str, case_accessions_train), map(str, review_accessions_train)
            control_accessions, case_accessions, review_accessions = map(str, control_accessions), map(str,
                                                                                                       case_accessions), map(
                str, review_accessions)

            save_accession_sql_format(control_accessions_train,
                                      self.param.output_path + '/Radiologist_Accessions/control_accessions_train.txt')
            save_accession_sql_format(case_accessions_train,
                                      self.param.output_path + '/Radiologist_Accessions/case_accessions_train.txt')
            save_accession_sql_format(review_accessions_train,
                                      self.param.output_path + '/Radiologist_Accessions/review_accessions_train.txt')

            save_accession_sql_format(control_accessions,
                                      self.param.output_path + '/Algorithm_Accessions/control_accessions.txt')
            save_accession_sql_format(case_accessions,
                                      self.param.output_path + '/Algorithm_Accessions/case_accessions.txt')
            save_accession_sql_format(review_accessions,
                                      self.param.output_path + '/Algorithm_Accessions/review_accessions.txt')

            print "Controls detected by radiologist: " + str(count_controls_train)
            print "Cases detected by radiologist: " + str(count_cases_train)
            print "Controls detected by algorithm: " + str(count_controls_apply)
            print "Cases detected by algorithm: " + str(count_cases_apply)

        else:
            for index in range(model_applied_reports.shape[0]):
                prediction = model_applied_reports['Confidence Category'].iloc[index]
                if prediction == self.confidence_labels[0]:
                    control_accessions.append(model_applied_reports['Accession1'].iloc[index])
                    count_controls_apply += 1

                elif prediction == self.confidence_labels[2]:
                    case_accessions.append(model_applied_reports['Accession1'].iloc[index])
                    count_cases_apply += 1
                else:
                    review_accessions.append(model_applied_reports['Accession1'].iloc[index])

            control_accessions, case_accessions, review_accessions = map(str, control_accessions), map(str,
                                                                                                       case_accessions), map(
                str, review_accessions)

            save_accession_sql_format(control_accessions,
                                      self.param.output_path + '/Full_Data_Applied_Accessions/control_accessions.txt')
            save_accession_sql_format(case_accessions,
                                      self.param.output_path + '/Full_Data_Applied_Accessions/case_accessions.txt')
            save_accession_sql_format(review_accessions,
                                      self.param.output_path + '/Full_Data_Applied_Accessions/review_accessions.txt')

            print "Controls detected by algorithm: " + str(count_controls_apply)
            print "Cases detected by algorithm: " + str(count_cases_apply)


def calc_metrics(true, prediction):
    cm = confusion_matrix(true, prediction)
    TN, FP, FN, TP = confusion_matrix(true, prediction).ravel()
    # print(classification_report(true, prediction))
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_Score = 2 * Precision * Recall / (Precision + Recall)

    return Accuracy, Precision, Recall, F1_Score, cm


def save_accession_sql_format(accessions, name):
    f = open(name, "w")

    out = ""
    for i, num in enumerate(accessions):
        if i % 1000 != 0:
            out += "\'" + num + "\', "
        else:
            out = out[:-2]
            out += "\n"
            f.write(out)
            out = ""

    out = out[:-2]
    out += "\n"
    f.write(out)

    f.close()


def main():
    parser = get_parser_classify()
    param = parser.parse_args()
    read = Radiology_Report_NLP(param=param)
    output_path_original = param.output_path

    for i_ngrams in range(len(param.max_ngrams)):
        if len(param.max_ngrams)>1:
            read.param.output_path = os.path.join(output_path_original, 'max_ngrams_'+str(param.max_ngrams[i_ngrams]))
            if not os.path.isdir(read.param.output_path):
                os.mkdir(read.param.output_path)
        read.i_ngrams = i_ngrams
        #
        read.preprocessing_full()

        model_default = "Best XGB"

        if not read.param.apply_all:
            read.run_models()
            #
            df_metrics_res = pd.read_csv(read.param.output_path + '/Metrics_Table.csv')
            print "Validation metrics of the trained models: "
            print df_metrics_res
            possible_model_names = [str(model) for model in df_metrics_res.columns[1:]]
            display_model_names = 'Choose between the following models: ' + ', '.join(possible_model_names) + '\n--> '
            chosen_model = raw_input(display_model_names)

            if chosen_model not in possible_model_names:
                print "ERROR: wrong model name"
                chosen_model = model_default
                # chosen_model = 'RandomForest'
                print "Using \"" + chosen_model + "\" as default model."

            assert chosen_model in read.dic_model_names.keys(), "ERROR: chosen model not in the model dictionary"
            # chosen_model='RandomForest'

        else:
            chosen_model = model_default

    if not param.no_apply:
        df_pred_apply = read.apply_model(model_name=chosen_model)
        df_pred_apply.to_csv(os.path.join(read.param.output_path, 'prediction_apply.csv'))

    # read.get_group_list_accessions(df_pred_apply)


if __name__ == "__main__":
    main()
