import pandas as pd
from ketiML.training import LSTM_model as LSTM_m
from ketiTF.general_transformation import data_scaling
from ketiTF.trans_for_purpose import trans_for_LSTMLearning as LDS
from pandas.tseries.frequencies import to_offset
# ## Inference Data Preprocessing

def inference(data_set, VI):

    DFS = data_scaling.DataFrameScaling(data_set, VI.scale_method)
    scale_columns = DFS.scale_columns
    scaler = DFS.set_scaler(VI.scaler_file_name)
    test = DFS.scaling_dataset(data_set)
    
    learning_information = VI.learning_information
    #data_frequency_sec = pd.to_timedelta(to_offset(pd.infer_freq(data_set[:5].index)), errors='coerce').total_seconds()
    TestDataSet = LDS.LearningDataSet(learning_information)
    test_X,learning_information= TestDataSet.get_inference_X(test)
    LSTM_P = LSTM_m.Predict(VI.model_file_name)
    LSTM_P.load_model()
    yhat = LSTM_P.predict(test_X, verbose_num=1)
    
    DIS = data_scaling.DataInverseScaling(VI.scale_method, scale_columns, VI.target_feature, DFS.scaler)
    inv_yhat = DIS.get_inv_Scaling_data(yhat)
    return inv_yhat[0]


if __name__ == "__main__":

    ## 0.예측에 필요한 적절한 모델과 전처리를 선택하기 위한 파라미터 셋팅
    # test_parameters에 주요 파라미터를 기술해놓음
    # 현재 파라미터는 80분 동안의 데이터를 바탕으로 12분 이후의 CO2ppm 값을 예측함 
    from settings import vibe_setting as vls
    from settings import test_parameters as ts
    VI= vls.vibe_learning(ts.features, ts.time_min, ts.learning_method_num)
    #########################

    ## 1. 예측에 필요한 입력 데이터
    ## 실제 VIBE 모듈에서는 입력 데이터를 dataframe 형태로 만들어 입력해야함
    import os
    dirname = os.path.dirname(__file__)
    test_data_fileName = os.path.join(dirname, 'data','test_data.csv')
    data_set = pd.read_csv(test_data_fileName, index_col=['datetime'], parse_dates=['datetime']) 

    data_set = data_set[VI.feature_list]
    data_set = data_set[-VI.X_row_num:]
    print(data_set)
    ######################################
    ## 2. inference
    # inference 함수에 data와 파라미터를입력하면 결과값을 전달함
    
    inv_yhat= inference(data_set, VI)  
    print(inv_yhat)