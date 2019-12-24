import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut, GridSearchCV, KFold


colours = ['pink', 'green', 'red', 'blue', 'black', 'yellow', 'purple', 'grey']
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p']

def add_list_multi(list0flist):
    outlist = list0flist[0]
    n = len(list0flist)
    for i in list0flist[1:]:
        outlist = list(map(add, outlist, i))

    outList = [x / n for x in outlist]
    return outList


def get_index(index_list):
    return nun_NaN_allsamp_per_df1['Peaks'][index_peak]


df1 = "/Users/Vartika_Bisht/Desktop/MSc. Bioinformatics/Module 3 Data Analytics & Statistical Machine Learning/COURSEWORK/normalizedtina.csv"
RNAseqInfo = pd.read_csv(df1)

Time1 = pd.DataFrame()
Time2 = pd.DataFrame()
Time3 = pd.DataFrame()
Time = [Time1, Time2, Time3]

# Time Data Frame list
for i in ['A', 'B']:
    for j in range(1, 5):
        for k in range(1, 4):
            col = i.__add__('R{}'.format(j)).__add__('T{}'.format(k))
            index = i.__add__('R{}'.format(j))
            Time[k - 1][index] = RNAseqInfo[col]

# Statistical test wrt Time
sampA = []
sampB = []
for t in range(1, 4):
    print('Time-{} Analysis :'.format(t))
    stA = stats.f_oneway(Time[t - 1]['AR1'], Time[t - 1]['AR2'], Time[t - 1]['AR3'], Time[t - 1]['AR4'])
    stB = stats.f_oneway(Time[t - 1]['BR1'], Time[t - 1]['BR2'], Time[t - 1]['BR3'], Time[t - 1]['BR4'])

    print('For sample A in Time-{} the F-test gives : '.format(t), stA)
    print('Means for Sample A Replicates are : ', statistics.mean(Time[t - 1]['AR1']),
          statistics.mean(Time[t - 1]['AR2']), statistics.mean(Time[t - 1]['AR3']), statistics.mean(Time[t - 1]['AR4']))
    print('Standard Deviation for Sample A Replicates are : ', statistics.stdev(Time[t - 1]['AR1']),
          statistics.stdev(Time[t - 1]['AR2']), statistics.stdev(Time[t - 1]['AR3']),
          statistics.stdev(Time[t - 1]['AR4']))

    print('For sample B in Time-{} the F-test gives : '.format(t), stB)
    print('Means for Sample B Replicates are : ', statistics.mean(Time[t - 1]['BR1']),
          statistics.mean(Time[t - 1]['BR2']), statistics.mean(Time[t - 1]['BR3']), statistics.mean(Time[t - 1]['BR4']))
    print('Standard Deviation for Sample B Replicates are : ', statistics.stdev(Time[t - 1]['BR1']),
          statistics.stdev(Time[t - 1]['BR2']), statistics.stdev(Time[t - 1]['BR3']),
          statistics.stdev(Time[t - 1]['BR4']))
    addA = list(map(add, list(map(add, Time[t - 1]['AR1'], Time[t - 1]['AR2'])),
                    list(map(add, Time[t - 1]['AR3'], Time[t - 1]['AR4']))))
    addB = list(map(add, list(map(add, Time[t - 1]['BR1'], Time[t - 1]['BR2'])),
                    list(map(add, Time[t - 1]['BR3'], Time[t - 1]['BR4']))))
    sampA.append(addA)
    sampB.append(addB)
    stAB = stats.f_oneway(sampA[t - 1], sampB[t - 1])
    print('For Time-{} the F-test between A and B gives : '.format(t), stAB)
    print()

# Within Each time wrt each sample
for t in range(1, 4):
    for samp in ['A', 'B']:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.boxplot(
            [Time[t - 1]['{}R1'.format(samp)], Time[t - 1]['{}R2'.format(samp)], Time[t - 1]['{}R3'.format(samp)],
             Time[t - 1]['{}R4'.format(samp)]])
        # ax1.scatter(range(25135), Time[t-1]['{}R1'.format(samp)], s=10, c='blue', marker="s", label='First Replicate')
        # ax1.scatter(range(25135), Time[t-1]['{}R2'.format(samp)], s=10, c='green', marker="x", label='Second Replicate')
        # ax1.scatter(range(25135), Time[t-1]['{}R3'.format(samp)], s=10, c='red', marker="o", label='Third Replicate')
        # ax1.scatter(range(25135), Time[t-1]['{}R4'.format(samp)], s=10, c='pink', marker=r'$\clubsuit$', label='Fourth Replicate')
        # plt.legend(loc='upper left');
        plt.title(label='Sample {}'.format(samp).__add__(' in Time-{}'.format(t)))
        plt.savefig('Box Plot Sample {}'.format(samp).__add__(' in Time-{}'.format(t)).__add__('RNAseq.png'))

# Time Series Analysis
TSA_A = []
TSA_B = []

for i in range(3):
    TSA_A_list = []
    TSA_B_list = []
    for j in Time[i].columns:
        if (j[0] == 'A'):
            TSA_A_list.append(Time[i][j])
        else:
            TSA_B_list.append(Time[i][j])
    TSA_A.append(add_list_multi(TSA_A_list))
    TSA_B.append(add_list_multi(TSA_B_list))

TSA_Diff = []
for i in range(3):
    TSA_Diff.append(list(map(sub, TSA_A_list[i], TSA_B_list[i])))

print(TSA_Diff)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.boxplot(TSA_Diff)

print("f-test for Time Series for A :", stats.f_oneway(TSA_A[0], TSA_A[1], TSA_A[2]))
print()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.boxplot(TSA_A)
plt.title("Box Plot of A in Time")
plt.savefig("Box Plot of A in Time RNAseq.png")

print("f-test for Time Series for B :", stats.f_oneway(TSA_B[0], TSA_B[1], TSA_B[2]))
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.boxplot(TSA_B)
plt.title("Box Plot of B in Time")
plt.savefig("Box Plot of B in Time RNAseq.png")
print()

fig = plt.figure()
ax1 = fig.add_subplot(111)
for j, i in enumerate(TSA_A[1:]):
    ax1.scatter(range(25135), i, s=10, c=colours[j], marker=markers[j], label='Time {}'.format(j + 2))
plt.title(label='Sample A Time Series')
plt.legend();
plt.savefig('Sample A Time Series RNAseq T2T3.png')

fig = plt.figure()
ax1 = fig.add_subplot(111)
for j, i in enumerate(TSA_B[1:]):
    ax1.scatter(range(25135), i, s=10, c=colours[j], marker=markers[j], label='Time {}'.format(j + 2))
plt.title(label='Sample B Time Series')
plt.legend();
plt.savefig('Sample B Time Series RNAseq T2T3.png')

# Sample A vs B
ABana = [add_list_multi(TSA_A), add_list_multi(TSA_B)]

gene_difference = pd.DataFrame()
gene_difference['gene_index'] = range(25135)
gene_difference['difference'] = list(map(sub, ABana[0], ABana[1]))

fig = plt.figure()
ax1 = fig.add_subplot(111)
for j, i in enumerate(TSA_B[1:]):
    ax1.scatter(range(25135), gene_difference['difference'])
plt.title(label='Sample A and B Difference')
plt.legend();
# plt.savefig('Sample B Time Series RNAseq T2T3.png')

gene_difference = gene_difference.sort_values(by=['difference'])
print(gene_difference.head(10))
gene_difference = gene_difference.sort_values(by=['difference'], ascending=False)
print(gene_difference.head(10))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.boxplot(ABana)
plt.title("Box Plot of A and B")
plt.savefig("Box Plot of A and B RNAseq.png")

print("f-test for A vs B : ", stats.f_oneway(ABana[0], ABana[1]))

fig = plt.figure()
ax1 = fig.add_subplot(111)
for j, i in enumerate(['A', 'B']):
    ax1.scatter(range(25135), ABana[j], s=10, c=colours[j], marker=markers[j], label='Sample '.__add__(i))
plt.title(label='Sample A vs Sample B')
plt.legend();
plt.savefig('Sample A vs Sample B RNAseq.png')

print("Being A or B will determine more differnce than the time")


#Random Forest
df3 = "/Users/Vartika_Bisht/Desktop/rna_log10_CNN_new.csv"
RF_dataframe = pd.read_csv(df3)
print(RF_dataframe)
print(RF_dataframe.columns)
num_col = len(RF_dataframe.columns)
X = RF_dataframe.drop(labels=['c','t3','t1','t2'], axis=1)
y = RF_dataframe[['c','t3','t1','t2']]

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

