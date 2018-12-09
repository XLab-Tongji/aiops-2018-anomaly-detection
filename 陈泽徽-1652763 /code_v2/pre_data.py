import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
from sklearn.preprocessing import StandardScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings
warnings.filterwarnings("ignore")

path = "data/"
train_data = pd.read_csv(path+'train.csv')
#test_data = pd.read_csv(path+'test.csv')
kpi_group = list(set(train_data["KPI ID"]))
for i in kpi_group:
    print("KPI ID ",i,"length :",len(train_data[train_data['KPI ID']==i]))


def compute_features(kpi_ID):
    type_data_1 = train_data[train_data["KPI ID"]== kpi_ID]
    #type_data_1_test = test_data[test_data['KPI ID'] == kpi_ID]

    train_x_1 = np.array(type_data_1['value'])
    #test_x = np.array(type_data_1_test['value'])
    train_y_1 = np.array(type_data_1['label'])
    #test_y = np.array(type_data_1_test['label'])

    # MR / MR_diff (not complete)

    def get_MR_value(data, win_size):
        order = (0, win_size)
        tempModel = sm.tsa.ARMA(data, order).fit()
        delta = tempModel.fittedvalues
        return delta

    def get_MR_Diff(MR_list):
        former_diff = []
        later_diff = []
        for index, value in enumerate(MR_list):
            if index - 1 < 0:
                former_diff.append(0)
            else:
                former_diff.append(value-MR_list[index-1])
            if index + 1 > MR_list.shape[0] - 1:
                later_diff.append(0)
            else:
                later_diff.append(MR_list[index+1]-value)
        return np.concatenate([np.array(former_diff).reshape(-1,1),np.array(later_diff).reshape(-1,1)], axis=1)
        
    def get_MR_features(data, win_size_list, diff=False):
        feature_arr = []
        for win_size in win_size_list:
            temp = get_MR_value(data, win_size).reshape(-1,1)
            if diff:
                pass
            if feature_arr == []:
                feature_arr = temp
            else:
                feature_arr = np.concatenate([feature_arr, temp], axis=1)
        return feature_arr

    def get_EWMA_feature(data, alpha_list):
        data_pd = pd.DataFrame(data)
        new_feature = []
        for each_alpha in alpha_list:
            temp = data_pd.ewm(alpha=each_alpha).mean()
            #print(temp.shape)
            if type(new_feature) == list:
                new_feature = temp
            else:
                new_feature = pd.concat([new_feature, temp], axis=1)
        return np.array(new_feature)

    def get_2rd_exp_smooth(data, alpha, beta):
        data_len = data.shape[0]
        s_1_pre = data[0]
        s_2_pre = data[0]
        exp_smooth_list = []
        for i in range(data_len):
            s_1 = alpha * data[i] + (1 - alpha) * s_1_pre
            s_2 = alpha * s_1 + (1 - alpha) * s_2_pre
            a = 2 * s_1 - s_2
            b = alpha * (s_1 - s_2) / (1 - alpha) 
            f = a + beta * b
            exp_smooth_list.append(f)
        return np.array(exp_smooth_list).reshape(-1,1)

    def get_ExpSmooth_feature(data, alpha_list, beta_list):
        feature_arr = []
        len_alpha = len(alpha_list)
        len_beta = len(beta_list)
        if len_alpha != len_beta:
            return feature_arr
        else:
            for alpha, beta in zip(alpha_list, beta_list):
                temp = get_2rd_exp_smooth(data, alpha, beta)
                #print(temp.shape)
                if type(feature_arr) == list:
                    feature_arr = temp
                else:
                    feature_arr = np.concatenate([feature_arr, temp], axis=1)
        return feature_arr

    # get the diff with forward (stride > 0) and backward (stride < 0)
    def get_diff(data, stride):
        data_len = data.shape[0]
        diff_list = []
        if stride > 0:
            for i in range(data_len):
                if i < stride:
                    diff_list.append(0)
                else:
                    diff_list.append(data[i] - data[i-stride])
        else:
            for i in range(data_len):
                if i >= data_len + stride:    # notice that stride here is a negative value
                    diff_list.append(0)
                else:
                    diff_list.append(data[i] - data[i-stride])
        return np.array(diff_list).reshape(-1,1)

    def get_diff_feature(data, stride_list):
        feature_arr = []
        len_stride = len(stride_list)
        for stride in stride_list:
            temp = get_diff(data, stride)
            #print(temp.shape)
            if type(feature_arr) == list:
                feature_arr = temp
            else:
                feature_arr = np.concatenate([feature_arr, temp], axis=1)
        return feature_arr

    # extract HoltWinter features

    def get_hw_feature(data):
        es_model = ExponentialSmoothing(data).fit()
        data_pd = pd.DataFrame(data)
        new_feature = es_model.predict(start=data_pd.index[0], end=data_pd.index[-1])
        return np.array(new_feature).reshape(-1, 1)

    #feature_MA = get_MR_features(train_x_1, [5, 10, 20])
    feature_EWMA = get_EWMA_feature(train_x_1, [0.1, 0.3, 0.5, 0.7, 0.9])
    feature_ExpSmooth = get_ExpSmooth_feature(train_x_1, [0.2,0.4,0.6,0.8],[0.4,0.6,0.4,0.2])
    feature_diff = get_diff_feature(train_x_1, [-100, -50, -20, -10, -5, -1, 1, 5, 10, 20, 50, 100])
    #feature_HW = get_hw_feature(train_x_1)
    
    #### test set ####
    #test_EWMA = get_EWMA_feature(test_x, [0.1, 0.3, 0.5, 0.7, 0.9])
    #test_ExpSmooth = get_ExpSmooth_feature(test_x, [0.2,0.4,0.6,0.8],[0.4,0.6,0.4,0.2])
    #test_diff = get_diff_feature(test_x, [-100, -50, -20, -10, -5, -1, 1, 5, 10, 20, 50, 100])

    all_feature = train_x_1.reshape(-1,1)
    all_feature = np.concatenate([all_feature, feature_EWMA], axis=1)
    all_feature = np.concatenate([all_feature, feature_ExpSmooth], axis=1)
    all_feature = np.concatenate([all_feature, feature_diff], axis=1)
    #all_feature = np.concatenate([all_feature, feature_HW], axis=1)
    
    '''
    test_feature = test_x.reshape(-1,1)
    test_feature = np.concatenate([test_feature, test_EWMA], axis=1)
    test_feature = np.concatenate([test_feature, test_ExpSmooth], axis=1)
    test_feature = np.concatenate([test_feature, test_diff], axis=1)
    
    kpi_train_size = all_feature.shape[0]
    train_test_concate = np.concatenate([all_feature, test_feature], axis=0)
    ss = StandardScaler()
    ss_data = ss.fit_transform(train_test_concate)
    
    kpi_train_data = ss_data[:kpi_train_size]
    kpi_test_data = ss_data[kpi_train_size:]
    '''
    ss = StandardScaler()
    kpi_train_data = ss.fit_transform(all_feature)

    print(kpi_train_data.shape)

    x_y = np.concatenate([kpi_train_data, train_y_1.reshape(-1,1)], axis=1)
    #x_test = np.concatenate([kpi_test_data, test_y.reshape(-1,1)], axis=1)

    x_y_pd = pd.DataFrame(x_y)
    #x_test_pd = pd.DataFrame(x_test)
    
    if not os.path.exists('feature_data'):
        os.mkdir('feature_data')
    #if not os.path.exists('test_data'):
    #    os.mkdir('test_data')
        
    x_y_pd.to_csv('feature_data/' + str(kpi_ID) + '.csv', index=False)
    #x_test_pd.to_csv('test_data/' + str(kpi_ID) + '.csv', index=False)

kpi_ID = input("input the type ID: (all/specify one):")

if kpi_ID == 'all':
    for each_kpi in kpi_group:
        compute_features(each_kpi)
else:
    compute_features(kpi_ID)

