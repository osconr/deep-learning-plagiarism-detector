###################################################################################################
# Setup and shared package dependencies                                                           #
###################################################################################################

# Other packages 

import numpy as np
import pandas as pd
import config

# Tensorflow

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence

###################################################################################################
# Data generating                                                             
###################################################################################################

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGeneratorSiamese(Sequence):
    """
    Data Generator for the Koppel Blog Corpus for the Baseline_using_MGBD notebook.
    """
    'Generates data for keras'
    def __init__(self, df, tokenizer, max_len=300,  batch_size= 32, num_clases = 2, shuffle = False):
        #self.file = file
        self.max_len = max_len
        self.batch_size = batch_size
        self.df = df #pd.read_csv(open(file))
        self.indices = self.df.index.to_list()
        self.num_classes = num_clases
        self.shuffle = shuffle
        self.tokenizer = tokenizer

        #self.on_epoch_end()

    def __len__(self):
        return len(self.indices)//self.batch_size

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indices of the batch
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]  # Select the list of IDs to return as batch
        batch = [self.indices[k] for k in index]  # Generate data
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        """
        This function gets a list of selected indices to form selected batch. A transformation is performed to create X and Y.
        X is the cosine similarity between the embedded paragraphs P1 and P2.
        :param batch: selected indices for df
        :return: batch of data
        """
        # == signifies, same author, != signifies diff author. flipping to observe the difference if any
        y = ((self.df.loc[batch]['author_1'] != self.df.loc[batch]['author_2']).astype(int)).to_numpy() # labels
        p1_column = self.df.loc[batch]['para1_text'].values
        p2_column = self.df.loc[batch]['para2_text'].values
        p1_embed = self.tokenizer.texts_to_sequences(p1_column)
        p2_embed = self.tokenizer.texts_to_sequences(p2_column)
        p1_embed = pad_sequences(p1_embed, maxlen=self.max_len, padding='post')
        p2_embed = pad_sequences(p2_embed, maxlen=self.max_len, padding='post')
        #print(p1_embed[0].shape, p2_embed[0].shape)
        #X = np.array([np.squeeze(cosine_similarity([p1_embed[i]], [p2_embed[i]])) for i in range(len(p1_embed))]) # cosine_similarity(p1_embed, p2_embed)
        #print(X.shape)
        # to_categorical(y, num_classes=self.num_classes)
        return [p1_embed, p2_embed], y

###################################################################################################
# Tokenizer and embeddings                                                                        #
###################################################################################################

def get_tokenizer(training_data, max_words):
    combined = training_data["para1_text"] + " " + training_data["para2_text"]
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(combined.values) 
    tokenizer.word_index = {e:i for e, i in tokenizer.word_index.items() if i <= max_words} # FIX
    return tokenizer

def get_embed_matrix(tokenizer, embedding_dim):
    embeddings_index = {}
    if embedding_dim == 50:
        f = open(config.config_io.get("embedding")) 
    elif embedding_dim == 100:
        f = open(config.config_io.get("embedding_100d")) 
    else:
        print('No such embedding')
        return False
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    embed_matrix = get_glove_embed_matrix(tokenizer, embeddings_index, embedding_vector_size = embedding_dim)
    print(embed_matrix.shape)  
    return embed_matrix

def get_glove_embed_matrix(t, embeddings_index, embedding_vector_size = 50):
    """
    t: tokenizer

    """
    not_present_list = []
    vocab_size = len(t.word_index) + 1
    embedding_matrix = np.random.rand(vocab_size, len(embeddings_index['no'])) # Fix: Random ini instead
    for word, i in t.word_index.items():
        embedding_vector = None
        if word in embeddings_index.keys():
            embedding_vector = embeddings_index.get(word)
        else:
            not_present_list.append(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embedding_matrix   

###################################################################################################
# Distance measures                                                                               #
###################################################################################################

def exponent_neg_cosine_distance(left,right):
    left = tf.keras.backend.l2_normalize(left, axis=-1)
    right = tf.keras.backend.l2_normalize(right, axis=-1)
    return tf.keras.backend.exp(tf.keras.backend.sum(tf.keras.backend.prod([left, right], axis=0), axis=1, keepdims=True))

def manh_lstm_distance(left, right):
    distance = tf.keras.backend.abs(left-right)
    distance = tf.keras.backend.sum(distance, axis=1, keepdims=True)
    distance = -distance
    distance = tf.keras.backend.exp(distance)
    return distance

def exponent_neg_euclidean_distance(left, right):
    distance = tf.keras.backend.square(left-right)
    distance = tf.keras.backend.sum(distance, axis=1, keepdims=True)
    distance = tf.keras.backend.sqrt(distance, axis = 1, keepdims = True)
    distance = tf.keras.backend.exp(-distance)
    return distance

##############################################################################################
# Measures
###############################################################################################

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

##############################################################################################
# Preprocessing data
##############################################################################################



def get_processed_dataset(datasetname):
    return pd.read_csv(config.processed_datasets[datasetname])

def preprocess_for_siamese(training_dataname, max_len, embedding_dim, batch_size, max_words=10000, training_prop=0.8):

    MAX_WORDS = max_words
    TRAINING_PROP = training_prop

    training_data = get_processed_dataset(training_dataname)
    tokenizer = get_tokenizer(training_data, max_words=MAX_WORDS)
    embed_matrix = get_embed_matrix(tokenizer, embedding_dim=embedding_dim)
    #print(embed_matrix)

    training_data_siamese = DataGeneratorSiamese(training_data.iloc[0:int(TRAINING_PROP*len(training_data))], tokenizer=tokenizer, max_len=max_len, batch_size=batch_size, shuffle=True)
    validation_data_siamese = DataGeneratorSiamese(training_data.iloc[int(TRAINING_PROP*len(training_data)):], tokenizer=tokenizer, max_len=max_len, batch_size=batch_size, shuffle=True)
    training_data_siamese.on_epoch_end() # this is a hack for "'DataGenerator' object, as has no attribute 'index'". 
    validation_data_siamese.on_epoch_end() # this is a hack for "'DataGenerator' object, as has no attribute 'index'". 

    return tokenizer, embed_matrix, training_data_siamese, validation_data_siamese

##############################################################################################
# Key model running script
##############################################################################################

def run_model(chosen_datasetname,
              custom_modelname,
              model_fun,
              optimizer,
              BATCH_SIZE = 64,
              EPOCHS = 50,
              STEPS_PER_EPOCH = 500,
              SHUFFLE = True,
              max_len = 300,
              embedding_dim = 50,
              early_stopping = False,
             verbose = 1,
             max_words=10000):

    """
    This function needs to be provided with a name, dataset and model function
    and it will run the model and save the history and checkpoints.
    
    @Params:
    chosen_datasetname - Dataset to use
    custom_modelname - History and checkpoints will be saved under this name, should be in format: model_fun + "_" + chosen_datasetname
    model_fun - Model to be created
    optimizer - Which optimiser the model uses
    BATCH_SIZE - - Corresponds to standard Keras function
    EPOCHS - Corresponds to standard Keras function
    STEPS_PER_EPOCH - Corresponds to standard Keras function
    SHUFFLE - Shuffles the data at the start of each epoch?!
    max_len - Maximum number of words to be included in embedding 
    embedding_dim - Number of embedding dimensions used
    early_stopping - If true then model stops once validation steps becomes constant
    
    Select the model and the optimiser by inputting them as 'model_fun' and 'optimizer'.
    """
    
    
    # Settings

    MAX_LEN = max_len
    EMBEDDING_DIM = embedding_dim
    VERBOSE = verbose
    MAX_WORDS = max_words

    # Get tokenizer, embedding matrix, and data
    tokenizer, embed_matrix, training_data, validation_data = preprocess_for_siamese(chosen_datasetname, MAX_LEN, EMBEDDING_DIM, BATCH_SIZE,max_words=MAX_WORDS)

    model = model_fun(max_len=MAX_LEN,
                        tokenizer=tokenizer, 
                        embed_matrix=embed_matrix,
                        embedding_dim=EMBEDDING_DIM,
                        optimizer=optimizer)

    # Checkpoints
    checkpoint_path = config.checkpoints.get("dir")+custom_modelname+config.checkpoints.get("name")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=VERBOSE, 
        save_freq=5*STEPS_PER_EPOCH,
        save_weights_only=True
        )

    model.save_weights(checkpoint_path.format(epoch=0))

    
    stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    if early_stopping is True:
        CALLBACKS = [cp_callback,stop_callback]
    else:
        CALLBACKS = [cp_callback]
    
    # Fit
    history = model.fit(training_data,
                    validation_data=validation_data,
                    verbose=VERBOSE,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    shuffle=SHUFFLE,
                    callbacks=CALLBACKS)

    # Plots

    acc = pd.DataFrame({"training acc":history.history['acc'],
                            "validation acc": history.history['val_acc']})
    acc_plot = acc.plot()
    fig1 = acc_plot.get_figure()
    fig1.savefig(config.base_dir+'plots/'+custom_modelname+'_acc.png')

    loss = pd.DataFrame({"training loss":history.history['loss'],
                            "validation loss": history.history['val_loss']})
    loss_plot = loss.plot()
    fig2 = loss_plot.get_figure()
    fig2.savefig(config.base_dir+'plots/'+custom_modelname+'_loss.png')

    # History

    history_path = config.history['dir']+custom_modelname+'.csv'
    pd.DataFrame.from_dict(history.history).to_csv(history_path,index=False)
    
##############################################################################################