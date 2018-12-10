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

from sklearn.utils import shuffle, resample

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


def over_sampling(X, y_):
    sm = SMOTE(ratio=0.5, k_neighbors=2, random_state=42)
    return sm.fit_resample(X, y_)


def isolaion_forest(X_train, outlier_fraction, X_outliers, labels, model_id):
    outliers_fraction = outlier_fraction

    i_forest = IsolationForest(n_estimators=10, behaviour='new', contamination=outliers_fraction, max_features=19,
                               random_state=42, max_samples=256)

    start_time = time.time()
    i_forest.fit(X_train)
    end_time = time.time()
    joblib.dump(i_forest, str(model_id) + '_i_forest')
    print('model building time:', end_time - start_time)

    start_time = time.time()

    y_pred = i_forest.predict(X_outliers)

    s = i_forest.decision_function(X_outliers)  # sklearn_score

    for i in range(len(y_pred)):
        if s[i] < -0.11:
            y_pred[i] = 1
        if s[i] > 0.075:
            y_pred[i] = -1

    end_time = time.time()
    print('data predict time:', end_time - start_time)

    y_pred = np.subtract(labels.reshape(1, -1)[0], y_pred)

    fn_s_mean, fn_s_min, fn_s_max = np.mean(s[y_pred == 0]), np.min(s[y_pred == 0]), np.max(s[y_pred == 0])
    fp_s_mean, fp_s_min, fp_s_max = np.mean(s[y_pred == 1]), np.min(s[y_pred == 1]), np.max(s[y_pred == 1])
    tn_s_mean, tn_s_min, tn_s_max = np.mean(s[y_pred == -1]), np.min(s[y_pred == -1]), np.max(s[y_pred == -1])
    tp_s_mean, tp_s_min, tp_s_max = np.mean(s[y_pred == 2]), np.min(s[y_pred == 2]), np.max(s[y_pred == 2])

    false_negatives = len(y_pred[y_pred == 0])
    false_positives = len(y_pred[y_pred == 1])
    true_negatives = len(y_pred[y_pred == -1])
    true_positives = len(y_pred[y_pred == 2])
  
    accuracy = (true_negatives + true_positives) / (
            (true_negatives + true_positives) + (false_negatives + false_positives))
    positives = true_positives + false_negatives
    precision = (true_positives + 1e-10) / (false_positives + true_positives + 1e-10)
    recall = (true_positives + 1e-10) / (positives + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall)
    output_str = str('model id: %s, accuracy: %.4f, f1 score: %.4f, precision: %.4f, recall: %.4f' \
                     % (model_id, accuracy, f1_score, precision, recall))
    output_csv = str('%s, %.4f, %.4f, %.4f, %.4f' % ('all', accuracy, f1_score, precision, recall))
    output_score = str('%s, %.4f, %.4f, %.4f\n%s, %.4f, %.4f, %.4f\n%s, %.4f, %.4f, %.4f\n%s, %.4f, %.4f, %.4f\n' % (
        'false_negatives', fn_s_mean, fn_s_min, fn_s_max, 'false_positives', fp_s_mean, fp_s_min, fp_s_max,
        'true_negatives', tn_s_mean, tn_s_min, tn_s_max, 'true_positives', tp_s_mean, tp_s_min, tp_s_max))
    with open('if_result.txt', 'a') as f:
        f.write(output_str + '\n')
    with open('if_output.csv', 'a') as f:
        f.write(output_csv + '\n')
    with open('if_score.txt', 'a') as f:
        f.write(output_score)


def isolaion_forest_main():

    N_FEATURES = 19

    data = pd.read_csv('IF-based.csv')

    data.drop(['0'], axis=1, inplace=True)
    data = shuffle(data)
    data.columns = [str(i) for i in range(20)]

    for i in range(1):
        x = data.drop(['14'], axis=1)

        y = data['14']
        print(set(y))
        x = np.array(x).reshape([-1, N_FEATURES])
        y = np.array(y).reshape(-1, 1)
        X_train, X_valid, y_train, y_valid = model_selection.train_test_split(x, y, test_size=0.2, random_state=0)
        X_train, y_train = over_sampling(X_train, y_train)

        outlier_fraction = len(y_train[y_train == 1]) / len(y_train)

        isolaion_forest(X_train, outlier_fraction, X_valid, y_valid, str(i))


if __name__ == '__main__':
    isolaion_forest_main()
