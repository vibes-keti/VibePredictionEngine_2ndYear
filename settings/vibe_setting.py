import learning_setting

def vibe_learning(features, time_min, learning_method_num):
   VI = learning_setting.VIBES(features, time_min, learning_method_num)    
   VI.set_parameter()
   VI.set_file_name()
   VI.learning_parameterSet()
   return VI


