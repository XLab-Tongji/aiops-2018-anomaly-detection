'''
@author: Lenovo
'''
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from pandas import Series

from sklearn import model_selection

from sklearn.ensemble import IsolationForest

from imblearn.over_sampling import SMOTE
from sklearn.externals import joblib

from sklearn.utils import shuffle,resample

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
def feature_normalization(X):
    std = StandardScaler()
    for i in X.columns:
        X[i] = Series(list(std.fit_transform(np.array(X[i]).reshape(-1, 1))))
    return X

def over_sampling(X,y_):
    sm = SMOTE(ratio=0.3, k_neighbors=2, random_state=42)
    return sm.fit_resample(X, y_)

def isolaion_forest(X_train,X_outliers,labels,model_id):
    outliers_fraction=len(X_outliers)/(len(X_outliers)+len(X_train))

    i_forest=IsolationForest(n_estimators=10,behaviour='new',contamination=outliers_fraction,max_features=4,random_state=42)

    start_time = time.time()
    i_forest.fit(X_train)
    end_time = time.time()
    joblib.dump(i_forest,str(model_id) + '_i_forest')
    print('model building time:', end_time - start_time)

    start_time = time.time()

    y_pred = i_forest.predict(X_outliers)
    end_time = time.time()
    print('data predict time:', end_time - start_time)

    y_pred = np.subtract(labels.reshape(1, -1)[0], y_pred)
    np.savetxt(model_id+"_if_y_pred.txt", y_pred, fmt="%d", delimiter=",")
    s=i_forest.decision_function(X_outliers)
    fn_s_mean,fn_s_min,fn_s_max=np.mean(s[y_pred == 0]),np.min(s[y_pred == 0]),np.max(s[y_pred == 0])
    fp_s_mean,fp_s_min,fp_s_max=np.mean(s[y_pred == 1]),np.min(s[y_pred == 1]),np.max(s[y_pred == 1])
    tn_s_mean, tn_s_min, tn_s_max = np.mean(s[y_pred == -1]),np.min(s[y_pred == -1]),np.max(s[y_pred == -1])
    tp_s_mean, tp_s_min, tp_s_max = np.mean(s[y_pred == 2]),np.min(s[y_pred == 2]),np.max(s[y_pred == 2])
    false_negatives = len(y_pred[y_pred == 0])
    false_positives = len(y_pred[y_pred == 1])
    true_negatives = len(y_pred[y_pred == -1])
    true_positives = len(y_pred[y_pred == 2])

    # print(y_pred)
    accuracy = (true_negatives + true_positives) / (
            (true_negatives + true_positives) + (false_negatives + false_positives))
    positives = true_positives + false_negatives
    precision = (true_positives + 1e-10) / (false_positives + true_positives + 1e-10)
    recall = (true_positives + 1e-10) / (positives + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall)
    output_str = str('model id: %s, accuracy: %.4f, f1 score: %.4f, precision: %.4f, recall: %.4f' \
                     % (model_id, accuracy, f1_score, precision, recall))
    output_csv = str('%s, %.4f, %.4f, %.4f, %.4f' % ('not all', accuracy, f1_score, precision, recall))
    output_score=str('%s, %.4f, %.4f, %.4f\n%s, %.4f, %.4f, %.4f\n%s, %.4f, %.4f, %.4f\n%s, %.4f, %.4f, %.4f\n'% ('false_negatives', fn_s_mean,fn_s_min,fn_s_max,'false_positives',fp_s_mean,fp_s_min,fp_s_max,'true_negatives',tn_s_mean, tn_s_min, tn_s_max,'true_positives',tp_s_mean, tp_s_min, tp_s_max))
    with open('if_result.txt', 'a') as f:
        f.write(output_str + '\n')
    with open('if_output.csv', 'a') as f:
        f.write(output_csv + '\n')
    with open('if_score.txt','a') as f:
        f.write(output_score)
def undersampling(train, abnormal_ratio):

    '''
    
    :param train: 
    :param abnormal_ratio: 
    :return: 
    '''
    # Get the indices per target value
    idx_0 = train[train.target == 0].index
    idx_1 = train[train.target == 1].index
    # Get original number of records per target value
    nb_0 = len(train.loc[idx_0])
    nb_1 = len(train.loc[idx_1])
    # Calculate the undersampling rate and resulting number of records with target=0
    undersampling_rate = ((1-abnormal_ratio)*nb_1)/(nb_0*abnormal_ratio)
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

def easyensemble(df, abnormal_ratio, n_subsets=10):
    train_resample = []
    for _ in range(n_subsets):
        sel_train = undersampling(df, abnormal_ratio)
        train_resample.append(sel_train)
    return train_resample

def append_type_feature(features_df,type_num):
    '''
    :param features_df: with original features
    :param type_num: kpi_num
    :return: with 4 more statical features and 1 type feature
    '''
    features_df.insert(len(features_df.columns),str(len(features_df.columns)),type_num)
    value_data=pd.read_csv('../anomaly-data/test.csv')
    series=value_data[value_data['KPI ID']==num_2_kpi[type_num]]['value']
    temps = pd.DataFrame(series.values)
    window = temps.expanding()
    features_df = pd.concat([features_df,window.min(), window.mean(), window.max(), temps.shift(-1)], axis=1)
    return features_df

def isolaion_forest_main():
    '''
    :return:test_data
    '''
    kpi_id = num_2_kpi[1]
    data =append_type_feature(pd.read_csv('../feature_data' + '/' + kpi_id + '.csv'),1)
    for i in range(2,27):
        kpi_id = num_2_kpi[i]
        data = data.append(append_type_feature(pd.read_csv('../feature_data' + '/' + kpi_id + '.csv'),i),ignore_index=True)

    N_FEATURES=19
    data = data.fillna(0)
    resample(data,replace=True)
    
    test_data = data.sample(frac=0.05)
    train_data = data.append(test_data)
    train_data.drop_duplicates(keep=False,inplace=True)

    # undersample normal data
    N_SUBSETS=1
    train_data.rename(columns={'14':'target'}, inplace = True)
    ensemble_data=easyensemble(train_data, 0.03,n_subsets=N_SUBSETS)


    for i in range(N_SUBSETS):
        data = ensemble_data[i]
        print(data.shape[0])
        x = data.drop(['target'],axis=1)
        scaler = StandardScaler().fit(x)
        x = scaler.transform(x)


        y = data['target']
        x, y = over_sampling(x, y)
        x = np.array(x).reshape(-1,N_FEATURES )
        y = np.array(y).reshape(-1, 1)
        X_train, X_valid, y_train, y_valid= model_selection.train_test_split(x, y,test_size=0.2,random_state=0)


        print('KPI ID:', kpi_id)
        print('KPI samples: total=', len(x))
        print('normal samples: total=', y[y==0].shape[0], ' normal/total =', y[y==0].shape[0] / len(x))
        print('anomaly samples: total=', (len(x) -  y[y==0].shape[0]), ' anomaly/total =',
              (len(x) -  y[y==0].shape[0]) / len(x))
        print('train samples: total=', X_train.shape[0], 'train#/sample#=', X_train.shape[0] / len(x))
        print('validation samples  : total=', (X_valid.shape[0]),
              'validation#/sample#=', (X_valid.shape[0] ) / len(x), 'anomaly/normal=',
              len(y[y==1]) / len(y[y==0]))

        isolaion_forest(X_train,X_valid,y_valid,str(i))
    return test_data

def model_test(test_data,mode=0):
    start_time=time.time()

    if mode==1:
        kpi_id = num_2_kpi[12]
        test_data = pd.read_csv('../feature_data' + '/' + kpi_id + '.csv')
        for i in range(13, 14):
            kpi_id = num_2_kpi[i]
            test_data = test_data.append(pd.read_csv('../feature_data' + '/' + kpi_id + '.csv'), ignore_index=True)

    test_labels = np.array(test_data['14'])
    test_data_features=test_data.drop(columns=['14'])

    BATCH_SIZE=2000
    N_MODELS=1
    models = []
    for i in range(N_MODELS):
        models.append(joblib.load(str(i) + '_i_forest'))
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    true_positives = 0
    for batch in range(test_data.shape[0]//BATCH_SIZE):
        start_index=batch*BATCH_SIZE
        end_index=start_index+BATCH_SIZE
        data=test_data_features[start_index:end_index]
        labels=test_labels[start_index:end_index]
        y_preds = []
        for i in range(N_MODELS):
            print(i)
            y_preds.append(models[i].predict(data))
        y_pred=[0]*data.shape[0]
        votes=[[0]*N_MODELS]*data.shape[0]


        for i in range(data.shape[0]):
            for j in range(N_MODELS):
                if y_preds[j][i]== -1:
                    votes[i][j]=1
            if sum(votes[i])>N_MODELS//2:
                y_pred[i]=-1
            else:
                y_pred[i] = 1

        y_pred = np.subtract(labels.reshape(1, -1)[0], y_pred)

        false_negatives += len(y_pred[y_pred == 0])
        false_positives += len(y_pred[y_pred == 1])
        true_negatives += len(y_pred[y_pred == -1])
        true_positives += len(y_pred[y_pred == 2])
    end_time = time.time()
    print('data predict time:', end_time - start_time)

    accuracy = (true_negatives + true_positives) / (
            (true_negatives + true_positives) + (false_negatives + false_positives))
    positives = true_positives + false_negatives
    precision = (true_positives + 1e-10) / (false_positives + true_positives + 1e-10)
    recall = (true_positives + 1e-10) / (positives + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall)
    output_str = str('never trained not all, accuracy: %.4f, f1 score: %.4f, precision: %.4f, recall: %.4f' \
                     % (accuracy, f1_score, precision, recall))
    output_csv = str('never trained not all, %.4f, %.4f, %.4f, %.4f' % ( accuracy, f1_score, precision, recall))
    with open('if_result.txt', 'a') as f:
        f. write(output_str + '\n')
    with open('if_output.csv', 'a') as f:
        f.write(output_csv + '\n')

    print('batch: ', batch + 1)
    print('samples: total=', test_data.shape[0])
    print('anomalies: total=', test_data[test_labels == 1].shape[0])
    print('normals: total=', test_data[test_labels == 0].shape[0])

    print('results: true positives=', true_positives, 'true negatives=', true_negatives, 'false positives=', false_positives, 'false negatives=', false_negatives)

if __name__=='__main__':
    test_data=isolaion_forest_main()
    model_test(test_data)
   #df=pd.DataFrame([[1,2,3,4],[1,2,3,4]])
   #print(append_type_feature(df,5))
