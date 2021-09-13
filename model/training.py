from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import roc_auc_score, accuracy_score
import pandas as pd

class Model_Finder:

    def __init__(self):


        self.clf = RandomForestClassifier()


    def get_best_params_for_random_forest(self,train_x,train_y):
        # initializing with different combination of parameters
        self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                           "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

        #Creating an object of the Grid Search class
        self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5,  verbose=3)
        #finding the best parameters
        self.grid.fit(train_x, train_y)

        #extracting the best parameters
        self.criterion = self.grid.best_params_['criterion']
        self.max_depth = self.grid.best_params_['max_depth']
        self.max_features = self.grid.best_params_['max_features']
        self.n_estimators = self.grid.best_params_['n_estimators']

        #creating a new model with the best parameters
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                          max_depth=self.max_depth, max_features=self.max_features)
        # training the mew model
        self.clf.fit(train_x, train_y)

        return self.clf

    def get_best_model(self, train_x, train_y, test_x, test_y):

        self.random_forest = self.get_best_params_for_random_forest(train_x, train_y)
        self.prediction_random_forest = self.random_forest.predict_proba(
                                                                        test_x)  # prediction using the Random Forest Algorithm

        if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
            self.random_forest_score = accuracy_score((test_y), self.prediction_random_forest)

        else:
            self.random_forest_score = roc_auc_score((test_y), self.prediction_random_forest[:,1]
                                                     )  # AUC for Random Forest
            print(self.random_forest_score)

        return self.random_forest


