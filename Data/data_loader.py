import pandas as pd

class Data_Getter:
    def __init__(self):
        pass
    def get_data(self):

            self.data= pd.read_csv(r'C:\Users\manoj\Desktop\Machine_Learning_Heart_disease\Data\heart_disease.csv') # reading the data file

            return self.data


#
# data_getter = Data_Getter()
# data = data_getter.get_data()
# print(data.head())
