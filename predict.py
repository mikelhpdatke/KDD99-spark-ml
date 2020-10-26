from numpy.core.multiarray import result_type
import pandas as pd
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split
import pickle
# Must declare data_dir as the directory of training and test files
# data_dir="./datasets/KDD-CUP-99/"
data_dir = "./"
filename = 'model'
# load
loaded_model = pickle.load(open(filename, 'rb'))

predict_Data = pd.read_csv("kqout", header=None)

print("Transforming data")
# Categorize columns: "protocol", "service", "flag", "attack_type"
predict_Data[1], protocols = pd.factorize(predict_Data[1])
predict_Data[2], services = pd.factorize(predict_Data[2])
predict_Data[3], flags = pd.factorize(predict_Data[3])
# predict_Data[41], attacks = pd.factorize(predict_Data[41])
y_predict = loaded_model.predict(predict_Data)
# print(y_predict)

labels = ['normal', 'buffer_overflow', 'loadmodule', 'perl', 'neptune',
          'smurf', 'guess_passwd', 'pod', 'teardrop', 'portsweep',
          'ipsweep', 'land', 'ftp_write', 'back', 'imap', 'satan', 'phf',
          'nmap', 'multihop', 'warezmaster', 'warezclient', 'spy',
          'rootkit']
result = {}
for i in y_predict:
    if not (result.get(labels[i])):
        result[labels[i]] = 1
    else:
        result[labels[i]] += 1
print(result)
