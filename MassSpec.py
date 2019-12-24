import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut, GridSearchCV, KFold
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential
import scipy.stats as stats
import matplotlib.pyplot as plt
from operator import sub

df2 = "/Users/Vartika_Bisht/Desktop/df_scale.csv"
df1 = "/Users/Vartika_Bisht/Desktop/MSc. Bioinformatics/Module 3 Data Analytics & Statistical Machine Learning/COURSEWORK/peaks_matrix.csv"
peaks_matrix = pd.read_csv(df1)
Imputed_peaks_matrix = pd.read_csv(df2)
peaks_matrix_justvals = pd.read_csv(df1)
peaks_matrix_justvals.drop(['sample_name','treatment','time_point','replicate'], axis=1, inplace=True)
peaks_matrix_justvals_corr = peaks_matrix_justvals


num_NaN = peaks_matrix.isnull().sum().sum()
print('The number of NaN in the data frame : ',num_NaN)
num_NaN_per = (num_NaN/(1649*43))*100
print('Which is {}% of the total data'.format(num_NaN_per))
nun_NaN_allsamp = peaks_matrix_justvals.isnull().sum()
nun_NaN_allsamp_per = (nun_NaN_allsamp/43)*100
nun_NaN_allsamp_per_df1 = pd.DataFrame({'index':range(1649),'Peaks':peaks_matrix_justvals.columns,'Percentage of NaN':nun_NaN_allsamp_per})
def get_peak_using(index_peak):
    return nun_NaN_allsamp_per_df1['Peaks'][index_peak]
nun_NaN_allsamp_per_df = nun_NaN_allsamp_per_df1.sort_values(by=['Percentage of NaN'], ascending = False)

rubbish = []
for i in range(1649):
    if(nun_NaN_allsamp_per_df['Percentage of NaN'][i] > 38):
        rubbish.append(nun_NaN_allsamp_per_df['Peaks'][i])

print("These peaks have more than 40% data missing : Rubbish Peaks --> ",rubbish)
print("We will remove these peaks. There are {} rubbish peaks".format(len(rubbish)))
peaks_matrix_justvals.drop(rubbish,axis=1, inplace=True)
print("The number of NaN in the data frame now is :",peaks_matrix_justvals.isnull().sum().sum())
print()

print("Now we are in the position to do impute the missing data")
print("We impute the missing data using MissForest")
print("This is the data frame that we get after we impute the data")
Imputed_peaks_matrix.rename( columns={'Unnamed: 0':'IndexNum'}, inplace=True )
Imputed_peaks_matrix_transpose = Imputed_peaks_matrix.transpose()
print()

print("Now we will separate the data with respect to time")
Time1 = pd.DataFrame()
Time2 = pd.DataFrame()
Time3 = pd.DataFrame()
Time = [Time1, Time2, Time3]
replicate_upper = [7, 5, 8, 7, 8, 8]
RU,samp_index = 0,0
for ab in ['A','B']:
    for time_point in range(1,4):
        time_var = 'T{}'.format(time_point)
        for replicate in range(1, replicate_upper[RU]+1):
            replicate_var = 'R{}'.format(replicate)
            index_inTimedf = ab + replicate_var + time_var
            Time[time_point - 1][index_inTimedf] = Imputed_peaks_matrix_transpose[samp_index][1:]
            samp_index = samp_index + 1
            print('-',end="")
        RU = RU + 1
print()
print("The data is separated with respect to time")

print()
print("Statistical Analysis with respect to time")
print("The null hypothesis of the one way F test is that two or more groups have the same population mean")
print()
# Statistical test wrt Time
ftest_T1_A = stats.f_oneway(Time1['AR1T1'],Time1['AR2T1'],Time1['AR3T1'],Time1['AR4T1'],Time1['AR5T1'],Time1['AR6T1'],Time1['AR7T1'])
ftest_T1_B = stats.f_oneway(Time1['BR1T1'],Time1['BR2T1'],Time1['BR3T1'],Time1['BR4T1'],Time1['BR5T1'],Time1['BR6T1'],Time1['BR7T1'])
ftest_T2_A = stats.f_oneway(Time2['AR2T2'],Time2['AR2T2'],Time2['AR3T2'],Time2['AR4T2'],Time2['AR5T2'])
ftest_T2_B = stats.f_oneway(Time2['BR2T2'],Time2['BR2T2'],Time2['BR3T2'],Time2['BR4T2'],Time2['BR5T2'],Time2['BR6T2'],Time2['BR7T2'],Time2['BR8T2'])
ftest_T3_A = stats.f_oneway(Time3['AR3T3'],Time3['AR2T3'],Time3['AR3T3'],Time3['AR4T3'],Time3['AR5T3'],Time3['AR6T3'],Time3['AR7T3'],Time3['AR8T3'])
ftest_T3_B = stats.f_oneway(Time3['BR3T3'],Time3['BR2T3'],Time3['BR3T3'],Time3['BR4T3'],Time3['BR5T3'],Time3['BR6T3'],Time3['BR7T3'],Time3['BR8T3'])


print('Time 1 :')
print('F test for sample A :',ftest_T1_A)
print('F test for sample B :',ftest_T1_B)
print()

print('Time 2 :')
print('F test for sample A :',ftest_T2_A)
print('F test for sample B :',ftest_T2_B)
print()

print('Time 3 :')
print('F test for sample A :',ftest_T3_A)
print('F test for sample B :',ftest_T3_B)
print()

print("This shows that the Replicates are different among the Sample")
print()

colours = ['pink','green','red','blue','black','yellow','purple','grey']
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p']

fig = plt.figure()
ax1 = fig.add_subplot(111)
for j,i in enumerate([Time1['AR1T1'],Time1['AR2T1'],Time1['AR3T1'],Time1['AR4T1'],Time1['AR5T1'],Time1['AR6T1'],Time1['AR7T1']]):
    ax1.scatter(range(1596), i, s=10, c=colours[j], marker=markers[j], label='Replicate {}'.format(j+1))
plt.title(label = 'Sample A Time 1')
plt.legend();
plt.savefig('Sample A Time 1.png')

fig = plt.figure()
ax1 = fig.add_subplot(111)
for j,i in enumerate([Time1['BR1T1'],Time1['BR2T1'],Time1['BR3T1'],Time1['BR4T1'],Time1['BR5T1'],Time1['BR6T1'],Time1['BR7T1']]):
    ax1.scatter(range(1596), i, s=10, c=colours[j], marker=markers[j], label='Replicate {}'.format(j+1))
plt.title(label = 'Sample B Time 1')
plt.legend();
plt.savefig('Sample B Time 1.png')

fig = plt.figure()
ax1 = fig.add_subplot(111)
for j,i in enumerate([Time2['AR2T2'],Time2['AR2T2'],Time2['AR3T2'],Time2['AR4T2'],Time2['AR5T2']]):
    ax1.scatter(range(1596), i, s=10, c=colours[j], marker=markers[j], label='Replicate {}'.format(j+1))
plt.title(label = 'Sample A Time 2')
plt.legend();
plt.savefig('Sample A Time 2.png')

fig = plt.figure()
ax1 = fig.add_subplot(111)
for j,i in enumerate([Time2['BR2T2'],Time2['BR2T2'],Time2['BR3T2'],Time2['BR4T2'],Time2['BR5T2'],Time2['BR6T2'],Time2['BR7T2'],Time2['BR8T2']]):
    ax1.scatter(range(1596), i, s=10, c=colours[j], marker=markers[j], label='Replicate {}'.format(j+1))
plt.title(label = 'Sample B Time 2')
plt.legend();
plt.savefig('Sample B Time 2.png')

fig = plt.figure()
ax1 = fig.add_subplot(111)
for j,i in enumerate([Time3['AR3T3'],Time3['AR2T3'],Time3['AR3T3'],Time3['AR4T3'],Time3['AR5T3'],Time3['AR6T3'],Time3['AR7T3'],Time3['AR8T3']]):
    ax1.scatter(range(1596), i, s=10, c=colours[j], marker=markers[j], label='Replicate {}'.format(j+1))
plt.title(label = 'Sample A Time 3')
plt.legend();
plt.savefig('Sample A Time 3.png')

fig = plt.figure()
ax1 = fig.add_subplot(111)
for j,i in enumerate([Time3['BR3T3'],Time3['BR2T3'],Time3['BR3T3'],Time3['BR4T3'],Time3['BR5T3'],Time3['BR6T3'],Time3['BR7T3'],Time3['BR8T3']]):
    ax1.scatter(range(1596), i, s=10, c=colours[j], marker=markers[j], label='Replicate {}'.format(j+1))
plt.title(label = 'Sample B Time 3')
plt.legend();
plt.savefig('Sample B Time 3.png')

TSA_T1_A = add_list_multi([Time1['AR1T1'],Time1['AR2T1'],Time1['AR3T1'],Time1['AR4T1'],Time1['AR5T1'],Time1['AR6T1'],Time1['AR7T1']])
TSA_T1_B = add_list_multi([Time1['BR1T1'],Time1['BR2T1'],Time1['BR3T1'],Time1['BR4T1'],Time1['BR5T1'],Time1['BR6T1'],Time1['BR7T1']])
TSA_T2_A = add_list_multi([Time2['AR2T2'],Time2['AR2T2'],Time2['AR3T2'],Time2['AR4T2'],Time2['AR5T2']])
TSA_T2_B = add_list_multi([Time2['BR2T2'],Time2['BR2T2'],Time2['BR3T2'],Time2['BR4T2'],Time2['BR5T2'],Time2['BR6T2'],Time2['BR7T2'],Time2['BR8T2']])
TSA_T3_A = add_list_multi([Time3['AR3T3'],Time3['AR2T3'],Time3['AR3T3'],Time3['AR4T3'],Time3['AR5T3'],Time3['AR6T3'],Time3['AR7T3'],Time3['AR8T3']])
TSA_T3_B = add_list_multi([Time3['BR3T3'],Time3['BR2T3'],Time3['BR3T3'],Time3['BR4T3'],Time3['BR5T3'],Time3['BR6T3'],Time3['BR7T3'],Time3['BR8T3']])

TSA_A = [TSA_T1_A,TSA_T2_A,TSA_T3_A]
TSA_B = [TSA_T1_B,TSA_T2_B,TSA_T3_B]

for i in range(3):
    TSA_A_list = []
    TSA_B_list = []
    for j in Time[i].columns:
        if(j[0] == 'A'):
            TSA_A_list.append(Time[i][j])
        else:
            TSA_B_list.append(Time[i][j])
    TSA_A.append(add_list_multi(TSA_A_list))
    TSA_B.append(add_list_multi(TSA_B_list))

TSA_Diff = []
for i in range(3):
    TSA_Diff.append(list(map(sub,TSA_A_list[i],TSA_B_list[i])))

print(TSA_Diff)
print(get_peak_using(TSA_Diff[0].index(max(TSA_Diff[0]))),get_peak_using(TSA_Diff[0].index(min(TSA_Diff[0]))))
print(get_peak_using(TSA_Diff[1].index(max(TSA_Diff[1]))),get_peak_using(TSA_Diff[1].index(min(TSA_Diff[1]))))
print(get_peak_using(TSA_Diff[2].index(max(TSA_Diff[2]))),get_peak_using(TSA_Diff[2].index(min(TSA_Diff[2]))))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.boxplot(TSA_Diff)
plt.show()

print("f-test for Time Series for A :",stats.f_oneway(TSA_T1_A,TSA_T2_A,TSA_T3_A))
print()
print("f-test for Time Series for B :",stats.f_oneway( TSA_T1_B,TSA_T2_B,TSA_T3_B))
print()

fig = plt.figure()
ax1 = fig.add_subplot(111)
for j,i in enumerate(TSA_A[:2]):
    ax1.scatter(range(1596),i , s=10, c=colours[j], marker=markers[j], label='Time {}'.format(j + 1))
plt.title(label = 'Sample A Time Series')
plt.legend();
plt.savefig('Sample A Time Series T1T2.png')

fig = plt.figure()
ax1 = fig.add_subplot(111)
for j,i in enumerate(TSA_B[:2]):
    ax1.scatter(range(1596),i , s=10, c=colours[j], marker=markers[j], label='Time {}'.format(j + 1))
plt.title(label = 'Sample B Time Series')
plt.legend();
plt.savefig('Sample B Time Series T1T2.png')


#Sample A vs B
ABana = [add_list_multi(TSA_A),add_list_multi(TSA_B)]

print("f-test for A vs B : ",stats.f_oneway(ABana[0],ABana[1]))

fig = plt.figure()
ax1 = fig.add_subplot(111)
for j,i in enumerate(['A','B']):
    ax1.scatter(range(1596),ABana[j] , s=10, c=colours[j], marker=markers[j], label='Sample '.__add__(i))
plt.title(label='Sample A vs Sample B')
plt.legend();
plt.savefig('Sample A vs Sample B.png')

#CNN
df3 = "/Users/Vartika_Bisht/Desktop/df_CNN.csv"
CNN_dataframe = pd.read_csv(df3)
print(CNN_dataframe)
print(CNN_dataframe.columns)
num_col = len(CNN_dataframe.columns)
X = np.asarray(CNN_dataframe.drop(labels = 'c',axis=1))
y = np.asarray(CNN_dataframe['c'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

neural_network = Sequential()
neural_network.add(Dense(activation = 'relu', input_dim = num_col - 1, units=6))
neural_network.add(Dense(activation = 'relu', units=6))
neural_network.add(Dense(activation = 'relu', units=6))
neural_network.add(Dense(activation = 'relu', units=6))
neural_network.add(Dense(activation = 'relu', units=6))
neural_network.add(Dense(activation = 'relu', units=6))

neural_network.add(Dense(activation = 'sigmoid', units=1))
neural_network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
neural_network.fit(X_train, y_train, batch_size = 5, epochs = 5)
y_pred = neural_network.predict(X_test)
y_pred = (y_pred > 0.5)
print(neural_network.get_config())

cm = confusion_matrix(y_test, y_pred)


#Random Forest
df3 = "/Users/Vartika_Bisht/Desktop/df_CNN.csv"
RF_dataframe = pd.read_csv(df3)
print(RF_dataframe)
print(RF_dataframe.columns)
num_col = len(RF_dataframe.columns)
X = RF_dataframe.drop(labels=['c', 't2'], axis=1)
y = RF_dataframe[['c', 't2']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

random_f_model = RandomForestClassifier(random_state=0)

kf = KFold(n_splits=5, random_state=15, shuffle=True)
count_k = 0
for train_index, test_index in kf.split(X):
    count_k += 1
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    parameters = {
        'n_estimators': [2, 3, 5],
        'max_depth': [1, 2, 3, 4],
        'min_samples_leaf': [2, 5, 10]
    }

    rf_grid_search = GridSearchCV(random_f_model, parameters, cv=5, scoring='f1_samples')
    grid_search = rf_grid_search.fit(X_train, y_train)

    best_random_f_model = rf_grid_search.best_estimator_
    y_test_predicted = best_random_f_model.predict(X_test)

    print(best_random_f_model.get_params())
    print(rf_grid_search.best_score_)

    for feature_name, feature_importance in zip(X_test.columns.values, best_random_f_model.feature_importances_):
        if feature_importance > 0.0:
            print('{:20s}:{:3.4f}'.format(feature_name, feature_importance))



#SVM
df3 = "/Users/Vartika_Bisht/Desktop/df_CNN.csv"
SVM_dataframe = pd.read_csv(df3)
num_col = len(SVM_dataframe.columns)
X = SVM_dataframe.drop(labels = ['t1','t2','t3'],axis=1)
y = SVM_dataframe[['t1','t2','t3']]

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

kf = KFold(n_splits=5, random_state=15, shuffle=True)
count_k = 0
for train_index, test_index in kf.split(X):
    count_k += 1
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-2, 1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5,
                       scoring=''score'')
    clf.fit(X_train, y_train)
    print(clf.best_params_)

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    model = svm.SVC(kernel=clf.best_params_['kernel'], gamma=clf.best_params_['gamma'], C=clf.best_params_['C'])
    model.fit(X, y)

plt.show()