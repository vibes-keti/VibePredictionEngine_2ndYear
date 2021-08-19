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
    data_frequency_sec = pd.to_timedelta(to_offset(pd.infer_freq(data_set[:5].index)), errors='coerce').total_seconds()
    TestDataSet = LDS.LearningDataSet(learning_information)
    test_X,learning_information= TestDataSet.get_inference_X(test)
    LSTM_P = LSTM_m.Predict(VI.model_file_name)
    LSTM_P.load_model()
    yhat = LSTM_P.predict(test_X, verbose_num=1)
    
    DIS = data_scaling.DataInverseScaling(VI.scale_method, scale_columns, VI.target_feature, DFS.scaler)
    inv_yhat = DIS.get_inv_Scaling_data(yhat)
    return inv_yhat[0]


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    sys.path.append("../..")

    import os
    dirname = os.path.dirname(__file__)
    test_data_fileName = os.path.join(dirname, 'data','test_data.csv')
    print(test_data_fileName)
    
    from KETIAppMachineLearning.VIBES.settings import vibe_setting as vls
    from KETIAppMachineLearning.VIBES.settings import test_parameters as ts

    VI= vls.vibe_learning(ts.features, ts.time_min, ts.learning_method_num)

     ## dataset preparation ##
    data_set = pd.read_csv(test_data_fileName, index_col=['datetime'], parse_dates=['datetime']) 
    data_set = data_set[VI.feature_list]
    data_set = data_set[-VI.X_row_num:]
    #########################
    inv_yhat= inference(data_set, VI)  
    print(inv_yhat)