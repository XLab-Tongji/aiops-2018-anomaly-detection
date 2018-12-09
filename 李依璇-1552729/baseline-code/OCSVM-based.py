'''
@author: Lenovo
'''
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import time
from sklearn.externals import joblib
from sklearn.utils import shuffle

num_2_kpi = {
    1: '09513ae3e75778a3',
    2: '769894baefea4e9e',
    3: '71595dd7171f4540',
    4: '18fbb1d5a5dc099d',
    5: '88cf3a776ba00e7c',
    6: 'a40b1df87e3f1c87',
    7: '40e25005ff8992bd',
    8: '046ec29ddf80d62e',
    9: '9bd90500bfd11edb',
    10: '9ee5879409dccef9',
    11: '8bef9af9a922e0b3',
    12: '07927a9a18fa19ae',
    13: '1c35dbf57f55f5e4',
    14: 'c58bfcbacb2822d1',
    15: 'affb01ca2b4f0b45',
    16: '02e99bd4f6cfb33f',
    17: '8a20c229e9860d0c',
    18: 'a5bf5d65261d859a',
    19: 'da403e4e3f87c9e0',
    20: '7c189dd36f048a6c',
    21: '54e8a140f6237526',
    22: 'e0770391decc44ce',
    23: 'cff6d3c01e6a6bfa',
    24: '8c892e5525f3e491',
    25: '76f4550c43334374',
    26: 'b3b2e6d1a791d63a'
}

def buildModel(mdl):
    n_features  =14
    models  ={ 'svm': SVC(C=0.1 ,kernel='rbf' ,gamma='scale' ,decision_function_shape='ovo'),
            'ocsvm': OneClassSVM(nu=0.1, kernel="rbf", gamma='scale')
            }
    return models[mdl]

def over_sampling(X,y_):
    sm = SMOTE(ratio=0.3, k_neighbors=2, random_state=42)
    return sm.fit_resample(X, y_)
def undersampling(train, desired_apriori):

    # Get the indices per target value
    idx_0 = train[train.target == 0].index
    idx_1 = train[train.target == 1].index
    # Get original number of records per target value
    nb_0 = len(train.loc[idx_0])
    nb_1 = len(train.loc[idx_1])
    # Calculate the undersampling rate and resulting number of records with target=0
    undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
    undersampled_nb_0 = int(undersampling_rate*nb_0)
    print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
    print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))
    # Randomly select records with target=0 to get at the desired a priori
    undersampled_idx = shuffle(idx_0,n_samples=undersampled_nb_0)
    #resample(undersampled_idx,n_samples=undersampled_nb_0,replace=True)
    # Construct list with remaining indices
    idx_list = list(undersampled_idx) + list(idx_1)
    # Return undersample data frame
    train = train.loc[idx_list].reset_index(drop=True)
    return train
def easyensemble(df, desired_apriori, n_subsets=10):
    train_resample = []
    for _ in range(n_subsets):
        sel_train = undersampling(df, desired_apriori)
        train_resample.append(sel_train)
    return train_resample

def ocsvm(X_train,X_outliers,labels,kpi_id):
    #oc_svm=buildModel('ocsvm')
    outliers_fraction = min(len(X_outliers)/len(X_train),0.99)
    oc_svm=OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma='scale')
    start_time = time.time()
    oc_svm.fit(X_train)
    end_time = time.time()
    joblib.dump(oc_svm,kpi_id+'oc_scm')
    print('model building time:', end_time - start_time)
    start_time = time.time()
    y_pred_outliers = oc_svm.predict(X_outliers)
    end_time = time.time()
    print('data predict time:', end_time - start_time)
    del oc_svm
    y_pred_outliers = np.subtract(labels.reshape(1, -1)[0], y_pred_outliers)
    np.savetxt("y_pred_outliers.txt", y_pred_outliers, fmt="%d", delimiter=",")
    '''
    true_positives = len(y_pred_outliers[y_pred_outliers == 0])
    true_negatives = len(y_pred_outliers[y_pred_outliers == 1])
    false_positives = len(y_pred_outliers[y_pred_outliers == -1])
    false_negatives = len(y_pred_outliers[y_pred_outliers == 2])
    '''

    false_negatives = len(y_pred_outliers[y_pred_outliers == 0])
    false_positives = len(y_pred_outliers[y_pred_outliers == 1])
    true_negatives = len(y_pred_outliers[y_pred_outliers == -1])
    true_positives = len(y_pred_outliers[y_pred_outliers == 2])

    # print(y_pred_outliers)
    accuracy = (true_negatives + true_positives) / (
            (true_negatives + true_positives) + (false_negatives + false_positives))
    positives = true_positives + false_negatives
    precision = (true_positives + 1e-10) / (false_positives + true_positives + 1e-10)
    recall = (true_positives + 1e-10) / (positives + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall)
    output_str = str('KPI ID: %s, accuracy: %.4f, f1 score: %.4f, precision: %.4f, recall: %.4f' \
                     % (kpi_id, accuracy, f1_score, precision, recall))
    output_csv = str('%s, %.4f, %.4f, %.4f, %.4f' % (kpi_id, accuracy, f1_score, precision, recall))
    with open('ocsvm_result.txt', 'a') as f:
        f.write(output_str + '\n')
    with open('ocsvm_output.csv', 'a') as f:
        f.write(output_csv + '\n')

def append_type_feature(features_df,type_num):
    features_df.insert(len(features_df.columns),str(len(features_df.columns)),type_num)
    value_data=pd.read_csv('../anomaly-data/test.csv')
    series=value_data[value_data['KPI ID']==num_2_kpi[type_num]]['value']
    temps = pd.DataFrame(series.values)
    window = temps.expanding()
    features_df = pd.concat([features_df,window.min(), window.mean(), window.max(), temps.shift(-1)], axis=1)
    print(features_df.head(5),features_df.columns)
    return features_df


def ocsvm_main():
    n_components=19
    simulator_production = 30 * 24 * 60
    N_SUBSETS = 1

    kpi_id = num_2_kpi[1]
    data = append_type_feature(pd.read_csv('../feature_data' + '/' + kpi_id + '.csv'), 1)
    data=data.fillna(0)
    test_data=data[simulator_production:-1].copy()
    test_data.rename(columns={'14':'target'},inplace=True)
    test_data=easyensemble(test_data,0.5,n_subsets=N_SUBSETS)[0]
    test_data.rename(columns={'target': '14'}, inplace=True)
    data=data[:simulator_production]

    for i in range(2, 5):
        kpi_id = num_2_kpi[i]
        data_tmp=append_type_feature(pd.read_csv('../feature_data' + '/' + kpi_id + '.csv'), i)
        data_tmp = data_tmp.fillna(0)
        if data_tmp.shape[0]>simulator_production:
            test_data_tmp = data_tmp[simulator_production:-1].copy()
            test_data_tmp.rename(columns={'14': 'target'}, inplace=True)
            test_data_tmp=easyensemble(test_data_tmp, 0.5, n_subsets=N_SUBSETS)[0]
            test_data_tmp.rename(columns={'target': '14'}, inplace=True)
            test_data=test_data.append(test_data_tmp,ignore_index=True)  # data[data['14']==1].copy()
            #print(test_data.shape,test_data_tmp.shape,i)
            data=data.append(data_tmp[:simulator_production],ignore_index=True)
            print(data.shape, data_tmp.shape, i)
        else:
             split = data_tmp.shape[0] // 2
             test_data_tmp = data_tmp[split:-1].copy()
             test_data_tmp.rename(columns={'14': 'target'}, inplace=True)
             test_data_tmp = easyensemble(test_data_tmp, 0.5, n_subsets=N_SUBSETS)[0]
             test_data_tmp.rename(columns={'target': '14'}, inplace=True)
             test_data=test_data.append(test_data_tmp,ignore_index=True)
             data=data.append(data_tmp[:simulator_production],ignore_index=True)
             print(data.shape, data_tmp.shape, i)




    #data= data.fillna(0)
    #test_data=test_data.fillna(0)
    resample(data, replace=True)
    print(data.shape)
    x= data.drop(['14'],axis=1)
    y = data['14']
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    #PCA doesn't improve traing time cost,try normalization
    print('feature_preprocessing completed..')
    x_inlier=np.array(x[y==0]).reshape(-1,n_components)
    y_inlier = np.array(y[y == 0]).reshape(-1, 1)

    X_train_inlier, X_valid_inlier, y_train_inlier, y_valid_inlier = model_selection.train_test_split(x_inlier, y_inlier, test_size=0.2, random_state=0)


    test_data_x=test_data.drop(['14'],axis=1)
    test_data_y = test_data['14']

    X_valid_outlier, y_valid_outlier = np.array(test_data_x).reshape(-1,n_components),np.array(test_data_y).reshape(-1,1)


    print('KPI ID:' ,kpi_id)
    print('KPI samples: total=',len(x))
    print('normal samples: total=', y_inlier.shape[0], ' normal/total =',y_inlier.shape[0]/len(x))
    print('anomaly samples: total=', (len(x)-y_inlier.shape[0]), ' anomaly/total =',(len(x)-y_inlier.shape[0])/len(x))
    print('train samples: total=',X_train_inlier.shape[0],'train#/sample#=',X_train_inlier.shape[0]/len(x))
    print('validation samples: total=',(test_data.shape[0]),'validation#/sample#=',(test_data.shape[0])/len(x),'anomaly/normal=',len(test_data[test_data_y==1])/len(test_data[test_data_y==0]))


    ocsvm(X_train_inlier,np.concatenate([X_valid_inlier,X_valid_outlier]),np.concatenate([y_valid_inlier,y_valid_outlier]),kpi_id)


if __name__=='__main__':
    ocsvm_main()











