base_dir = '/home/sukanya/PhD/CLEF/'

config_io = {'pan_21_processed_train': base_dir + 'Datasets/PAN SCD/pan21-style-change-detection/processed/train.csv',
             'feature_set_path': '/home/sukanya/PhD/Results/011_16_Apr_PAN_21_dataset/vocab/opt2/200word_list.txt',
             'pan_21_processed_test': base_dir + 'Datasets/PAN SCD/pan21-style-change-detection/processed/test.csv',
             'embedding': '/home/sukanya/PhD/Embeddings/Glove/glove.6B.50d.txt',
             'checkpoint_bilstm': "training_2/cp-{epoch:04d}.ckpt",
             'checkpoint_bigru': "training_gru/cp-{epoch:04d}.ckpt"}

checkpoints = {'checkpoint_bilstm_2021': "training_bilstm_2021/cp-{epoch:04d}.ckpt",
               'checkpoint_bilstm_2020': "training_bilstm_2020/cp-{epoch:04d}.ckpt",
               'checkpoint_bilstm_2019': "training_bilstm_2019/cp-{epoch:04d}.ckpt",
               'checkpoint_bilstm_2018': "training_bilstm_2018/cp-{epoch:04d}.ckpt",
               'checkpoint_bilgru_2021': "training_bilgru_2021/cp-{epoch:04d}.ckpt",
               'checkpoint_bilgru_2020': "training_bilgru_2020/cp-{epoch:04d}.ckpt",
               'checkpoint_bilgru_2019': "training_bilgru_2019/cp-{epoch:04d}.ckpt",
               'checkpoint_bilgru_2018': "training_bilgru_2018/cp-{epoch:04d}.ckpt"}

original_datasets = {'train_2017': base_dir+"2017/pan17-style-breach-detection-training-dataset-2017-02-15",
      'test_2017' : base_dir+"2017/pan17-style-breach-detection-test-dataset-2017-02-15",
      'train_2018': base_dir+"2018/pan18-style-change-detection-training-dataset-2018-01-31",
      'valid_2018': base_dir+"2018/pan18-style-change-detection-validation-dataset-2018-01-31",
      'test_2018': base_dir+"2018/pan18-style-change-detection-test-dataset-2018-01-31",
      'train_2019': base_dir+"2019/pan19-style-change-detection-training-dataset-2019-01-17",
      'valid_2019': base_dir+"2019/pan19-style-change-detection-validation-dataset-2019-01-17",
      'train_2020_wide': base_dir+"2020/data/pan20-style-change-detection/train/dataset-wide",
      'valid_2020_wide': base_dir+"2020/data/pan20-style-change-detection/validation/dataset-wide",
      'train_2020_narrow': base_dir+"2020/data/pan20-style-change-detection/train/dataset-narrow",
      'valid_2020_narrow': base_dir+"2020/data/pan20-style-change-detection/validation/dataset-narrow",
      'train_2021': base_dir + '2021/train',
      'valid_2021': base_dir + '2021/validation'}


pan21_format = {'train_2017': base_dir+"converted_data/2017/train",
      'test_2017' : base_dir+"converted_data/2017/test",
      'train_2018': base_dir+"converted_data/2018/train",
      'valid_2018': base_dir+"converted_data/2018/validation",
      'test_2018': base_dir+"converted_data/2018/test",
      'train_2019': base_dir+"converted_data/2019/train",
      'valid_2019': base_dir+"converted_data/2019/validation",
      'train_2020_wide': base_dir+"converted_data/2020/wide/train",
      'valid_2020_wide': base_dir+"converted_data/2020/wide/validation",
      'train_2020_narrow': base_dir+"converted_data/2020/narrow/train",
      'valid_2020_narrow': base_dir+"converted_data/2020/narrow/validation",
      'train_2021': base_dir + '2021/train',
      'valid_2021': base_dir + '2021/validation'}


processed ={'train_2017': base_dir+"processed_data/2017/train.csv",
      'test_2017' : base_dir+"processed_data/2017/test.csv",
      'train_2018': base_dir+"processed_data/2018/train.csv",
      'valid_2018': base_dir+"processed_data/2018/validation.csv",
      'test_2018': base_dir+"processed_data/2018/test.csv",
      'train_2019': base_dir+"processed_data/2019/train.csv",
      'valid_2019': base_dir+"processed_data/2019/validation.csv",
      'train_2020_wide': base_dir+"processed_data/2020/wide/train.csv",
      'valid_2020_wide': base_dir+"processed_data/2020/wide/validation.csv",
      'train_2020_narrow': base_dir+"processed_data/2020/narrow/train.csv",
      'valid_2020_narrow': base_dir+"processed_data/2020/narrow/validation.csv",
      'train_2021': base_dir + "processed_data/2021/train.csv",
      'valid_2021': base_dir + "processed_data/2021/validation.csv",
            }



