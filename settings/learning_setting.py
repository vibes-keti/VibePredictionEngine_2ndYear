import os
import datetime

class LearningInference():
   def __init__(self):
      self.target_data_preparation_method_list = ['step', 'mean', 'max', 'min']
      self.learning_style_list=['LSTM']
      self.learning_method_list = ['VanilaLSTM', 'StackedLSTM', 'BiDirectionalLSTM', 'CNNLSTM', 'ConvLSTM']
      self.dirname = os.path.dirname(__file__)
      self.scale_method_list = [['scale'], ['scale','log'],['log']]
      self.scale_method_num =0 
      self.target_method_num = 1
      self.scale_method =self.scale_method_list[self.scale_method_num] 
      self.target_data_preparation_method = self.target_data_preparation_method_list[self.target_method_num]

   def set_parameter(self):

      #################################################################
      # make other parameters automatically 
      self.learning_style =self.learning_style_list[self.learning_style_num]
      self.learning_method = self.learning_method_list[self.learning_method_num]
      self.n_features = len(self.feature_list)

      self.future_sec = self.future_min*60
      self.past_sec = self.past_min * 60
      self.X_row_num = int(self.past_min/self.re_frequency_min)
      self.re_frequency = datetime.timedelta(seconds= self.re_frequency_min*60)
      #################################################################

   def learning_parameterSet(self):
      self.val_length_ratio = 0.8
      self.learning_parameter = {
         "first_unit_num":64,
         "second_unit_num":64,
         "CNN_filter":64,
         "ConvLSTM2D":64
      }
      self.epochs_num = 50
      self.verbose_num=2
      ###
      self.learning_information= {
         "re_frequency_min":self.re_frequency_min,
         "future_num" : int(self.future_min/self.re_frequency_min),
         "past_num" : int(self.past_min/self.re_frequency_min),
         "learning_method":self.learning_method,
         "learning_style":self.learning_style,
         "target_feature":self.target_feature,
         "n_features":self.n_features,
         "learning_parameter":self.learning_parameter
      }

   def set_file_name(self):
      #scaler
      scaler_id ="Num_"+str(self.n_features)+'.pkl'
      self.scaler_file_name = os.path.join(self.dirname,'..', 'scaler',self.partial_dataset_type, str(self.feature_list), scaler_id)

      model_id = 'pastMin_'+str(self.past_min)+'futureMin_'+str(self.future_min)+'reMIn_'+str(self.re_frequency_min)
      self.model_file_name = os.path.join(self.dirname,'..', 'model', self.partial_dataset_type, self.target_feature, self.learning_method, str(self.feature_list), model_id)

class VIBES(LearningInference):
   def __init__(self, features, time_min, learning_method_num):
      super().__init__()
      # data cleansing
      self.data_type = 'air' # this parameter is used for feature based cleaning

      self.target_feature = features['target_feature'] 
      self.feature_list = features['feature_list']
      self.future_min = time_min['future_min']
      self.past_min = time_min['past_min']
      self.re_frequency_min= time_min['re_frequency_min']

      self.partial_dataset_type = 'farm1' 
      self.learning_method_num = learning_method_num
      self.learning_style_num=0






