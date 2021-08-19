#### List
target_feature_list = ['CO2ppm','H2Sppm','Humidity','Temperature','NH3ppm']
past_min_list = [80, 120, 720]
future_min_list = [12, 20, 60]
re_frequency_min_list = [4, 10]
all_features =['CO2ppm', 'H2Sppm', 'Humidity', 'NH3ppm', 'Temperature', 'out_CO',
               'out_NO2', 'out_O3', 'out_PM10', 'out_PM25', 'out_SO2', 'out_humid',
               'out_pressure', 'out_rainfall', 'out_sunshine', 'out_temp',
               'out_wind_direction', 'out_wind_speed']

####
target_feature ='CO2ppm'
features_list_set=[['CO2ppm'], ['CO2ppm','NH3ppm'],['CO2ppm','H2Sppm','Humidity','Temperature','NH3ppm'], 
['CO2ppm','NH3ppm','Humidity','Temperature','out_humid','out_temp'], all_features]

features_list= features_list_set[3]
features ={'target_feature':target_feature, "feature_list": features_list}
time_min={'past_min' :past_min_list[0], "future_min": future_min_list[0],"re_frequency_min":re_frequency_min_list[0] }
learning_method_num = 4
## CO2ppm ##
test_y_quantile_set=[600, 800]
""""""

"""
features_list=['CO2ppm', 'H2Sppm', 'Humidity', 'NH3ppm', 'Temperature','out_humid','out_temp', 'out_wind_speed']
features ={'target_feature':target_feature, "feature_list": features_list}
time_min={'past_min' :120, "future_min": 20,"re_frequency_min":4 }
learning_method_num = 0

#CO2ppm
test_y_quantile_set=[600, 800]
"""
"""
features_list=['CO2ppm', 'H2Sppm', 'Humidity', 'NH3ppm', 'Temperature']
features ={'target_feature':target_feature, "feature_list": features_list}
time_min={'past_min' :120, "future_min": 20,"re_frequency_min":4 }
learning_method_num = 4

features_list=['CO2ppm', 'H2Sppm', 'Humidity', 'NH3ppm', 'Temperature']
features ={'target_feature':target_feature, "feature_list": features_list}
time_min={'past_min' :720, "future_min": 60,"re_frequency_min":10 }
learning_method_num = 4

"""
