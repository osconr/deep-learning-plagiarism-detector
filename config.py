#base_dir = '/home/jovyan/examples/examples/tensorflow/project-2022-group-4/'
#base_dir = '/Users/asmusharre/Documents/GitHub/project-2022-group-4/'
base_dir = '/Users/conradosmond/Documents/MSc_ASDC_Repos/st456_seminars/project-2022-group-4/'

config_io = {
      'embedding': base_dir + 'embeddings/glove.6B.50d.txt',
      'embedding_100d': base_dir + 'embeddings/glove.6B.100d.txt'
}

checkpoints = {
      'dir': base_dir + "checkpoints/",
      'name': "/cp-{epoch:04d}.ckpt",
      'bilstm_2021': base_dir + "checkpoints/training_bilstm_2021/cp-{epoch:04d}.ckpt",
      'bilstm_2022': base_dir + "checkpoints/training_bilstm_2022/cp-{epoch:04d}.ckpt"
      }

history = {
      'dir': base_dir + "history/",
      'bilstm_2021': base_dir + "history/training_bilstm_2021.csv",
      'bilstm_2022': base_dir + "history/training_bilstm_2022.csv"
}

original_datasets = {
      'train_2021': base_dir + 'data/pan21/train',
      'valid_2021': base_dir + 'data/pan21/validation',
      'train_2022': base_dir + 'data/pan22/dataset1/train',
      'valid_2022': base_dir + 'data/pan22/dataset1/validation',
      'train_2022_task2': base_dir + 'data/pan22/dataset2/train',
      'valid_2022_task2': base_dir + 'data/pan22/dataset2/validation'
            }

processed_datasets = {
      'train_2021': base_dir + "processed/pan21/train.csv",
      'valid_2021': base_dir + "processed/pan21/validation.csv",
      'train_2022_task1': base_dir + "processed/pan22/dataset1/train.csv",
      'valid_2022_task1': base_dir + "processed/pan22/dataset1/validation.csv",
      'train_2022_task2': base_dir + "processed/pan22/dataset2/train.csv",
      'valid_2022_task2': base_dir + "processed/pan22/dataset2/validation.csv"
            }



