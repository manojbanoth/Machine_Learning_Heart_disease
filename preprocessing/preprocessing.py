import pandas as pd
from sklearn.preprocessing import StandardScaler

class preprocess:
    def __init__(self):
        pass


    def is_null_present( self, data):

        # self.null_counts = data.isna().sum()  # check for the count of null values per column
        # for i in self.null_counts:
        #     if i > 0:
        data.dropna(axis=0,inplace=True)
    #print(data.shape)

        return data



    def dropUnnecessaryColumns(self,data, columnNameList):

        data= data.drop(columnNameList, axis=1)
        return data

    def separate_label_feature(self,data, label_column_name):
        self.X = data.drop(columns=label_column_name,
                               axis=1)  # drop the columns specified and separate the feature columns
        self.Y = data[label_column_name]  # Filter the Label columns

        return self.X, self.Y

    def std_scalar(self, X):
        scalar = StandardScaler()
        X_scaled = scalar.fit_transform(X)

        return X_scaled

    # x = heart_df.drop(columns='target')
    # y = heart_df.target