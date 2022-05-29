import json
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def convertPAN_to_siamese_format_TIRA(problem_path, tokenizer):
    """
    No truth file should be accessible
    """
    with open(problem_path) as f:
        text = f.read()
    p1_list = []
    p2_list = []
    paragraphs = text.split("\n")
    non_zero_len_paragraphs = [para for para in paragraphs if len(para)>0] # problem-601 creates an empty paragraph causing the problem. so had to remove empty paragraphs
    for i in range(0, len(non_zero_len_paragraphs)-1):
        p1_list.append(paragraphs[i])
        p2_list.append(paragraphs[i+1])
    df = pd.DataFrame({'para1_text':p1_list,
                      'para2_text':p2_list})
    p1_column = df['para1_text'].values
    p2_column = df['para2_text'].values
    p1_embed = tokenizer.texts_to_sequences(p1_column)
    p2_embed = tokenizer.texts_to_sequences(p2_column)
    max_len = len(paragraphs)-1
    p1_embed = pad_sequences(p1_embed, maxlen=max_len, padding='post')
    p2_embed = pad_sequences(p2_embed, maxlen=max_len, padding='post')
    X = np.array([np.squeeze(cosine_similarity([p1_embed[i]], [p2_embed[i]])) for i in range(len(p1_embed))]) # cosine_similarity(p1_embed, p2_embed)
    return X

def convertPAN_to_siamese_format_TIRA_siamese(problem_path, tokenizer, max_len):
    """
    No truth file should be accessible
    """
    with open(problem_path) as f:
        text = f.read()
    p1_list = []
    p2_list = []
    paragraphs = text.split("\n")
    non_zero_len_paragraphs = [para for para in paragraphs if len(para)>0] # problem-601 creates an empty paragraph causing the problem. so had to remove empty paragraphs
    for i in range(0, len(non_zero_len_paragraphs)-1):
        p1_list.append(paragraphs[i])
        p2_list.append(paragraphs[i+1])
    df = pd.DataFrame({'para1_text':p1_list,
                      'para2_text':p2_list})
    p1_column = df['para1_text'].values
    p2_column = df['para2_text'].values
    p1_embed = tokenizer.texts_to_sequences(p1_column)
    p2_embed = tokenizer.texts_to_sequences(p2_column)
    #max_len = len(paragraphs)-1
    p1_embed = pad_sequences(p1_embed, maxlen=max_len, padding='post')
    p2_embed = pad_sequences(p2_embed, maxlen=max_len, padding='post')
    return [p1_embed, p2_embed]



def convertPAN_to_siamese_format(problem_path,truth_path, problem_file):
    with open(problem_path) as f:
        text = f.read()
    #print("len:", len(text))
    with open(truth_path) as json_file:
        data = json.load(json_file)
    a1_list = []
    a2_list = []
    p1_list = []
    p2_list = []
    p_list = []
    paragraph_authors = data['paragraph-authors']
    paragraphs = text.split("\n")
    non_zero_len_paragraphs = [para for para in paragraphs if len(para)>0] # problem-601 creates an empty paragraph causing the problem. so had to remove empty paragraphs
    if len(non_zero_len_paragraphs)!= len(paragraph_authors):
        print(f"Error! Number of paragraphs {len(non_zero_len_paragraphs)} does not equal paragraph authors {len(paragraph_authors)}! God help you now!")
        print(paragraph_authors)
        for para in non_zero_len_paragraphs:
            print(f"\n len: {len(para)} {para}")
            return None, None, None, None
    else:
        for i in range(0, len(paragraph_authors)):
            a1 = paragraph_authors[i]
            if i+1<len(paragraph_authors) and (i+1)< len(paragraphs):
                a2 = paragraph_authors[i+1]
                p1 = non_zero_len_paragraphs[i]
                p2 = non_zero_len_paragraphs[i+1]
                a1_list.append(a1)
                p1_list.append(p1)
                a2_list.append(a2)
                p2_list.append(p2)
                p_list.append(problem_file)
    df = pd.DataFrame({"problem":p_list, "author_1":a1_list,
                       "author_2":a2_list, "para1_text": p1_list, "para2_text":p2_list})
    return df


def write_solution(problem_file, solutions_folder, sol_dict):
    if not os.path.exists(solutions_folder):
        os.makedirs(solutions_folder)
    solution_file = solutions_folder+"/"+"solution-"+problem_file.split(".")[0]+".json"
    with open(solution_file, 'w+') as fp:
        json.dump(sol_dict, fp)#indent=4
    print("Writing in path:",solution_file )


def get_feature_set_files(path):
    feature_set_files = {}
    sub_folder_list = [name for name in os.listdir(path) if os.path.isdir(path)]
    sub_folder_list.sort()
    for sub_folder in sub_folder_list:
        vocab_files = os.listdir(path+sub_folder+"/")
        vocab_sizes = [int(i.split("w")[0]) for i in vocab_files]
        vocab_files_sorted = [x for _, x in sorted(zip(vocab_sizes, vocab_files))]
        vocab_files_sorted = [path+sub_folder+"/"+file for file in vocab_files_sorted]
        #vocab_files.sort(reverse = True)
        feature_set_files[sub_folder] = vocab_files_sorted
    return feature_set_files

