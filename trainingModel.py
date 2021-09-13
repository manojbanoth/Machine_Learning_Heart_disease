from sklearn.model_selection import train_test_split
from Data import data_loader
from preprocessing import preprocessing
from model import training
import pickle

class trainModel:
    print('0')




    def training_Model(self):
        print('1')

        data_getter = data_loader.Data_Getter()
        data = data_getter.get_data()
        print(data.head())

        """doing the data preprocessing"""
        print("3")
        preprocessor = preprocessing.preprocess()
        print("s")

        # check if missing values are present in the dataset
        data = preprocessor.is_null_present(data)
        print("N")
        print(data.shape)


        # removing unwanted columns as discussed in the EDA part in ipynb file
        data = preprocessor.dropUnnecessaryColumns(data, ['diabetes','diaBP','prevalentHyp','currentSmoker','BPMeds', 'prevalentStroke','BMI', 'heartRate','education'])
        print(data.head())
        # create separate features and labels
        X, Y = preprocessor.separate_label_feature(data,'TenYearCHD')

        # Convert into stnd scalar
        X_scaled = preprocessor.std_scalar(X)

        # splitting the data into training and test set
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=1 / 3,
                                                            random_state=355)

        model_finder = training.Model_Finder()  # object initialization
        # getting the best model
        best_model = model_finder.get_best_model(x_train, y_train, x_test, y_test)
        #print(best_model)
        filename = 'heart_disease.pkl'
        pickle.dump(best_model, open(filename, 'wb'))


a=trainModel()
a.training_Model()







