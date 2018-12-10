'''
@author: Lenovo
'''

import pandas as pd

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from pandas import Series
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import math
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

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
    n_features=14
    models={'lr':LogisticRegression(),
            'rf': RandomForestClassifier(n_estimators=10, max_features=int(math.sqrt(n_features)), max_depth=None, min_samples_split=2,
                              bootstrap=True)
            }
    return models[mdl]


M='rf'
def main():
    kpi_id = num_2_kpi[1]
    data = pd.read_csv('../feature_data' + '/' + kpi_id + '.csv')
    for i in range(2, 27):
        kpi_id = num_2_kpi[i]
        data = data.append(pd.read_csv('../feature_data' + '/' + kpi_id + '.csv'), ignore_index=True)


    std = StandardScaler()
    x = data.drop(['14'], axis=1)
    y = data['14']
    if M == 'lr':
        x=std.fit_transform(x)

    sm = SMOTE(ratio=1,k_neighbors=2,random_state=42)
    x,y=sm.fit_resample(x,y)

   
    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(x, y, test_size=0.2, random_state=0)
    if 1.0 not in set(y_valid):
        print(set(y_valid))
        print('error')



    clf = buildModel(M)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    cm = confusion_matrix(y_valid, y_pred, [1, 0])
    print(cm)
    true_positives = cm[0, 0]
    false_positives = cm[1, 0]
    true_negatives = cm[1, 1]
    false_negatives = cm[0, 1]
    accuracy = (true_negatives + true_positives) / (
                (true_negatives + true_positives) + (false_negatives + false_positives))
    positives = true_positives + false_negatives
    precision = (true_positives + 1e-10) / (false_positives + true_positives + 1e-10)
    recall = (true_positives + 1e-10) / (positives + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall)
    output_str = str('accuracy: %.4f, f1 score: %.4f, precision: %.4f, recall: %.4f' \
                     % (accuracy, f1_score, precision, recall))
    output_csv = str('%.4f, %.4f, %.4f, %.4f' % (accuracy, f1_score, precision, recall))
    with open(M+'_result.txt', 'a') as f:
        f.write(output_str + '\n')
    with open(M+'_output.csv', 'a') as f:
        f.write(output_csv + '\n')

'''
report=classification_report(y_valid, y_pred).splitlines(6)
print(report[0])
print(report[3])
'''

if __name__=='__main__':
    main()

