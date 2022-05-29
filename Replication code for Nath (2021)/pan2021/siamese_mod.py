#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
from spacy.lang.en import English
import spacy

spacy.prefer_gpu()

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Concatenate, GlobalMaxPool2D, Multiply
from tensorflow.keras.layers import Dropout, Subtract, Add, GlobalAvgPool2D, Conv2D, Bidirectional
from tensorflow.keras.layers import Flatten, Dense, Embedding, LSTM, GRU, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Concatenate, Lambda, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.initializers import glorot_uniform, Constant
from sklearn.model_selection import train_test_split
import os
import json
from DataGeneratorSiamese import DataGeneratorSiamese
from tqdm import tqdm
from processing import convertPAN_to_siamese_format_TIRA_siamese, write_solution


# In[2]:


def get_tokenizer(training_data, max_words):
    combined = training_data["para1_text"] + " " + training_data["para2_text"]
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(combined.values)
    return tokenizer


# In[3]:


def get_embed_matrix(tokenizer, embedding_dim):
    embeddings_index = {}
    f = open(config.config_io.get("embedding"))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    embed_matrix = get_glove_embed_matrix(tokenizer, embeddings_index, embedding_vector_size=embedding_dim)
    print(embed_matrix.shape)
    return embed_matrix


# In[4]:


def train_from_scratch(training_path, model_type="LSTM", max_words=10000, max_len=300,
                       embedding_dim=50, batch_size=64):
    print("Loading training file from path: ",
          config.config_io.get('pan_21_processed_train'))  # pan_20_processed_train_wide
    training_data = pd.read_csv(training_path)
    tokenizer = get_tokenizer(training_data, max_words)
    embed_matrix = get_embed_matrix(tokenizer, embedding_dim)
    len_train = len(training_data)
    training_generator = DataGeneratorSiamese(training_data.iloc[0:int(0.8 * len_train)], tokenizer=tokenizer,
                                              max_len=max_len, batch_size=64)
    validation_generator = DataGeneratorSiamese(training_data.iloc[int(0.8 * len_train):], tokenizer=tokenizer,
                                                max_len=max_len, batch_size=64)

    # this is a hack for "'DataGenerator' object has no attribute 'index'". It turns out that on_epoch_end creates the index that is used
    training_generator.on_epoch_end()
    validation_generator.on_epoch_end()
    if model_type == "GRU":
        model = create_gru_model(max_len=max_len,
                                 tokenizer=tokenizer,
                                 embed_matrix=embed_matrix,
                                 embedding_dim=embedding_dim)
        checkpoint_path = config.config_io.get("checkpoint_bigru")
    else:
        model = create_lstm_model(max_len=max_len,
                                  tokenizer=tokenizer,
                                  embed_matrix=embed_matrix,
                                  embedding_dim=embedding_dim)
        checkpoint_path = config.config_io.get("checkpoint_bilstm")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=5 * batch_size)
    model.save_weights(checkpoint_path.format(epoch=0))
    stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)  # val_loss
    history = model.fit(training_generator, validation_data=validation_generator, verbose=1, batch_size=batch_size,
                        epochs=60, steps_per_epoch=100,
                        callbacks=[cp_callback, stop_callback])  # , callbacks=[callback]
    return model


# In[16]:

def create_gru_model(max_len, tokenizer, embed_matrix, embedding_dim):
    input_1 = Input(shape=(max_len,))  # (train_p1_seq.shape[1],)
    input_2 = Input(shape=(max_len,))
    gru_layer = Bidirectional(GRU(units=50, dropout=0.2, recurrent_dropout=0.2))
    embeddings_initializer = Constant(embed_matrix)
    emb = Embedding(len(tokenizer.word_index) + 1,
                    embedding_dim,
                    embeddings_initializer=embeddings_initializer,
                    input_length=max_len,
                    weights=[embed_matrix],
                    trainable=True)
    e1 = emb(input_1)
    x1 = gru_layer(e1)
    e2 = emb(input_2)
    x2 = gru_layer(e2)
    distance = lambda x: exponent_neg_cosine_distance(x[0], x[1])  # manh_lstm_distance
    merged = Lambda(function=distance, output_shape=lambda x: x[0], name='cosine')([x1, x2])
    preds = Dense(1, activation='sigmoid')(merged)
    model = Model(inputs=[input_1, input_2], outputs=preds)
    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(clipnorm=1.5))
    print(model.summary())
    return model


def get_PAN_test_files(test_input_path):
    print(test_input_path)
    base = [file.split(".")[0] for file in os.listdir(test_input_path) if "txt" in file]
    problem_files = [file + ".txt" for file in base]
    return problem_files


# In[14]:


def execute_PAN_test(test_input_path, model, output_path, tokenizer, max_len):
    problem_files = get_PAN_test_files(test_input_path)
    for i in tqdm(range(0, len(problem_files))):
        x = convertPAN_to_siamese_format_TIRA_siamese(test_input_path + "/" + problem_files[i], tokenizer, max_len)
        print("\n", problem_files[i])
        # print(x)
        y_pred = model.predict(x, verbose=1)
        changes = np.where(y_pred > 0.5, 1, 0).flatten().tolist()
        multi_author = 0
        if sum(changes) >= 1:
            multi_author = 1
        paragraph_authors = get_cluster(changes)  # [0]*(len(changes)+1)
        sol_dict = {"multi-author": multi_author, "changes": changes,
                    "paragraph-authors": paragraph_authors}
        print(sol_dict)
        write_solution(problem_files[i], output_path, sol_dict)


# In[7]:


def execute(test_input_path, output_path, train_again=False, model_type="LSTM"):
    print("Model type:", model_type)
    max_len = 300
    max_words = 10000
    embedding_dim = 50
    # "BGRU"
    training_path = config.config_io.get('pan_21_processed_train')
    training_data = pd.read_csv(training_path)
    tokenizer = get_tokenizer(training_data, max_words)
    if train_again:
        model = train_from_scratch(training_path=training_path,
                                   model_type=model_type,
                                   max_len=max_len,
                                   max_words=max_words,
                                   embedding_dim=embedding_dim)
    else:
        embed_matrix = get_embed_matrix(tokenizer, embedding_dim)
        if model_type == "GRU":
            checkpoint_path = config.config_io.get("checkpoint_bigru")
            model = create_gru_model(max_len=max_len,
                                     tokenizer=tokenizer,
                                     embed_matrix=embed_matrix,
                                     embedding_dim=embedding_dim)
        else:
            checkpoint_path = config.config_io.get("checkpoint_bilstm")
            model = create_lstm_model(max_len=max_len,
                                      tokenizer=tokenizer,
                                      embed_matrix=embed_matrix,
                                      embedding_dim=embedding_dim)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
    execute_PAN_test(test_input_path=test_input_path,
                     model=model,
                     output_path=output_path,
                     tokenizer=tokenizer,
                     max_len=max_len)


def create_lstm_model(max_len, tokenizer, embed_matrix, embedding_dim):
    input_1 = Input(shape=(max_len,))  # (train_p1_seq.shape[1],)
    input_2 = Input(shape=(max_len,))
    lstm_layer = Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
    embeddings_initializer = Constant(embed_matrix)
    emb = Embedding(len(tokenizer.word_index) + 1,
                    embedding_dim,
                    embeddings_initializer=embeddings_initializer,
                    input_length=max_len,
                    weights=[embed_matrix],
                    trainable=True)

    e1 = emb(input_1)
    x1 = lstm_layer(e1)

    e2 = emb(input_2)
    x2 = lstm_layer(e2)

    distance = lambda x: exponent_neg_cosine_distance(x[0], x[1])  # manh_lstm_distance
    merged = Lambda(function=distance, output_shape=lambda x: x[0], name='cosine')([x1, x2])
    preds = Dense(1, activation='sigmoid')(merged)
    model = Model(inputs=[input_1, input_2], outputs=preds)
    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(clipnorm=1.5))
    print(model.summary())
    return model


# In[9]:


def get_glove_embed_matrix(t, embeddings_index, embedding_vector_size=50):
    """
    t: tokenizer

    """
    not_present_list = []
    vocab_size = len(t.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, len(embeddings_index['no'])))
    for word, i in t.word_index.items():
        embedding_vector = None
        if word in embeddings_index.keys():
            embedding_vector = embeddings_index.get(word)
        else:
            not_present_list.append(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.zeros(embedding_vector_size)  # size of the embedding
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embedding_matrix


# In[10]:


def exponent_neg_cosine_distance(left, right):
    left = tf.keras.backend.l2_normalize(left, axis=-1)
    right = tf.keras.backend.l2_normalize(right, axis=-1)
    return tf.keras.backend.exp(
        tf.keras.backend.sum(tf.keras.backend.prod([left, right], axis=0), axis=1, keepdims=True))


# In[11]:


def manh_lstm_distance(left, right):
    distance = tf.keras.backend.abs(left - right)
    distance = tf.keras.backend.sum(distance, axis=1, keepdims=True)
    distance = -distance
    distance = tf.keras.backend.exp(distance)
    return distance


# In[12]:


def exponent_neg_euclidean_distance(left, right):
    distance = tf.keras.backend.square(left - right)
    distance = tf.keras.backend.sum(distance, axis=1, keepdims=True)
    distance = tf.keras.backend.sqrt(distance, axis=1, keepdims=True)
    distance = tf.keras.backend.exp(-distance)
    return distance


# In[ ]:

def get_cluster(changes):
	cluster = [1]
	prev_author = 1
	for i,j in enumerate(changes):
		if j==0:
			curr_author = prev_author
			cluster.append(curr_author)
		else:
			curr_author = prev_author+1
			prev_author = curr_author
			cluster.append(curr_author)
	return cluster


# In[ ]:

'''
output_path = "/home/sukanya/PhD/CLEF/2021/runs/run_6_may_siamese_cosine_tira"
test_input_path = '/home/sukanya/PhD/Datasets/PAN SCD/pan21-style-change-detection/validation'
execute(test_input_path, output_path, train_again=False)

python3 main.py -c "/home/sukanya/PhD/CLEF/2021/train" -o "/home/sukanya/PhD/CLEF/2021/runs/run_14_may_1_siamese_LSTM_training" 2>&1 | tee lstm_training_output.txt
python3 main.py -c "/home/sukanya/PhD/CLEF/2021/validation" -o "/home/sukanya/PhD/CLEF/2021/runs/run_14_may_1_siamese_LSTM_validation" 2>&1 | tee lstm_validation_output.txt
python3 main.py -c "/home/sukanya/PhD/CLEF/2021/test_without_groundtruth" -o "/home/sukanya/PhD/CLEF/2021/runs/run_14_may_1_siamese_LSTM_test" 2>&1 | tee lstm_test_output.txt

python3 main.py -c "/home/sukanya/PhD/CLEF/2021/train" -o "/home/sukanya/PhD/CLEF/2021/runs/run_14__may_1_siamese_GRU_training" 2>&1 | tee gru_training_output.txt
python3 main.py -c "/home/sukanya/PhD/CLEF/2021/validation" -o "/home/sukanya/PhD/CLEF/2021/runs/run_14_may_1_siamese_GRU_validation" 2>&1 | tee gru_validation_output.txt
python3 main.py -c "/home/sukanya/PhD/CLEF/2021/test_without_groundtruth" -o "/home/sukanya/PhD/CLEF/2021/runs/run_14_may_1_siamese_LSTM_test" 2>&1 | tee gru_test_output.txt

python3 main.py -c "/home/sukanya/PhD/CLEF/2021/train" -o "/home/sukanya/PhD/CLEF/2021/runs/run_14_may_1_MFW_200_training" 2>&1 | tee mfw_training_output.txt
python3 main.py -c "/home/sukanya/PhD/CLEF/2021/validation" -o "/home/sukanya/PhD/CLEF/2021/runs/run_14_may_1_MFW_200_validation" 2>&1 | tee mfw_validation_output.txt
python3 main.py -c "/home/sukanya/PhD/CLEF/2021/test_without_groundtruth" -o "/home/sukanya/PhD/CLEF/2021/runs/run_14_may_1_MFW_200_test" 2>&1 | tee mfw_test_output.txt

python3 main.py -c "/home/sukanya/PhD/CLEF/2021/train" -o "/home/sukanya/PhD/CLEF/2021/runs/run_25_may_1_random_baseline_training" 2>&1 | tee random_training_output.txt
python3 main.py -c "/home/sukanya/PhD/CLEF/2021/validation" -o "/home/sukanya/PhD/CLEF/2021/runs/run_25_may_1_random_baseline_validation" 2>&1 | tee random_validation_output.txt
python3 main.py -c "/home/sukanya/PhD/CLEF/2021/test_without_groundtruth" -o "/home/sukanya/PhD/CLEF/2021/runs/run_25_may_1_random_baseline_test" 2>&1 | tee random_test_output.txt

python3 main.py -c "/home/sukanya/PhD/CLEF/converted_data/2020/wide/train" -o "/home/sukanya/PhD/CLEF/2021/runs/run_4_june_siamese_gru_training_2020_wide" 2>&1 | tee gru_training_2020_wide_output.txt

'''
