
########
#     _________________
#    |                 | 
#    | Package Upload  |
#    |_________________|
#    
########

# System packages
import json
import os
import time
import sys
from os import listdir
from os.path import isfile, join

# plotting packages
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
import math, scipy, copy, re
from bs4 import BeautifulSoup

import cgi

# used in the count of words
import string

# NLTK toolkit
import nltk
import nltk.data # natural language tool kit
# for tokenizing sentences according by the words
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize # $ pip install nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.cluster.util import cosine_distance

# Machine Learning Packages
import joblib
from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


# additional python files 
import model 
import sample
import encoder
#import forms

#from forms import ContactForm
from flask import render_template, jsonify, request, flash
from writrly import application
#from flask_mail import Message, Mail

###############################
#     ___________________________________
#    |                                   | 
#    |  Checks whether GPU is available  |
#    |___________________________________|
#    
################################

def GPU_avail():
    # Checking if GPU is available
    work_with_gpu = torch.cuda.is_available()
    if(work_with_gpu):
        return (" Program is using the GPU!")
    else: 
        return ('No GPU available. Using CPU.')


################################
#     _______________________
#    |                       | 
#    |  Code for Classifier  |
#    |_______________________|
#    
################################


## cleaning text
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text


## key for classifier
class_key = ['Business',
 'Education',
 'Entertainment',
 'Health',
 'Ideas',
 'International',
 'Politics',
 'Science',
 'Short Story',
 'Technology']

# load in csv file
masterDF = pd.read_csv('./writrly/static/data/master_df.csv')

#load model from file
file_model = joblib.load("./writrly/static/data/logreg_wordcount_model.pkl")

# clean essay element
essay = masterDF['essay']
essay = [clean_text(elem) for elem in essay]
masterDF['essay'] = essay

# create a count vectorizer object  and fit it to essa
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(masterDF['essay'])

# predicts class of string or file
def predict_class(free_x):
  
    #clean text
    free_x = clean_text(free_x)

    free_vect = count_vect.transform([free_x])

    prediction = class_key[file_model.predict(free_vect)[0]]

    return prediction, max(file_model.predict_proba(free_vect)[0])

def predict_class_file(filename):
  
    with open(filename, 'r') as file:
        free_x = file.read().replace('\n', '')  

    # clean text 
    free_x = clean_text(free_x)

    free_vect = count_vect.transform([free_x])

    prediction = class_key[file_model.predict(free_vect)[0]]

    return prediction, max(file_model.predict_proba(free_vect)[0])


################################
#     _____________________________
#    |                             |
#    | Code for Summary Extraction |
#    |_____________________________|
#    
################################


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in clean_text(sent1)]
    sent2 = [w.lower() for w in clean_text(sent2)]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary_text(text, top_n):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  sent_tokenize(text)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_matrix(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
        summarize_text.append(ranked_sentence[i][1])

    # Step 5 - Ofcourse, output the summarize texr
    #print("Summarize Text: \n",textwrap.fill(" ".join(summarize_text), 50))
    return(summarize_text)


################################
#     _____________________
#    |                     |
#    | Code for Generators |
#    |____________________ |
#    
################################

# for some reason requires double loading
import encoder

# function that removes ending punctuations
def remove_end_punct(string):
    reverse_string = string[::-1]
  
    i1 = reverse_string.find('.')
    i2 = reverse_string.find('?')
    i3 = reverse_string.find('!')
  
    if i1 == -1:
        i1 = 1000
    if i2 == -1:
        i2 = 1000
    if i3 == -1:
        i3 = 10000
    
    ifinal = min([i1, i2, i3])

    return string[:len(string)-ifinal]

## dictionary for model

## def print string given input
def simple_gen_str(input_string, lens, temp, model_choice):
    
    model_name=model_choice
    seed=None
    raw_text = '\n\n\n\n'+input_string
    length=lens
    temperature=temp #set to 1.0 for highest diversity
    top_k=40 #set to 40
    top_p=0.9 #set to 0.9
    
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    
    # produce only a single batch
    batch_size = 1

    # create encoder based on chosen model
    enc = encoder.get_encoder(model_name)
    
    # selects default hyperparameters based on model
    hparams = model.default_hparams()
    
    # overrides default hyperparameters with parameters from chosen model
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    # Cannot produce number of tokens more than the hyperparameters count
    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context, 
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        
        saver = tf.train.Saver()
        
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        
        # restores model from checkpoint
        saver.restore(sess, ckpt)
    
        # encodes input text
        context_tokens = enc.encode(raw_text)

        #runs session to generate encoded text given encoded raw text
        out = sess.run(output, feed_dict={context: 
                                          [context_tokens for _ in range(batch_size)]})[:, len(context_tokens):]
        
        #decodes output 
        text = enc.decode(out[0])

        # remove quadruple \n at beginning of text
        text = text.replace('\n\n\n\n', ' ')
        
        # remove triple \n at end of text
        text = text.replace('\n\n\n', '\n\n')
    
    return(text)

# with model chosen from text similarity or length 
def class_gen_str(input_text,  T, length, model_dict):
  
    # predicts class of input text
    old_class, old_prob = predict_class(input_text)

    if length == None:
        # text_length
        length = min([950, len(word_tokenize(input_text))])

    # return new text
    new_string =  simple_gen_str(input_string = input_text, 
              lens = length, 
              temp = T, 
              model_choice = model_dict[old_class])  

    # predicts probability of class of new string
    new_class, new_prob = predict_class(new_string)

    return remove_end_punct(new_string), [old_class, old_prob], [new_class, new_prob] 

# with model chosen from text similarity or length 
def class_extract_gen_str(input_text,  T, length, model_dict):
  
    # predicts class of input text
    old_class, old_prob = predict_class(input_text)

    if length == None:
        # text_length
        length = min([950, len(word_tokenize(input_text))])

    # summary list of sentences
    summ_list = generate_summary_text(text = input_text, top_n=3)
    
    # summary list of strings
    summ_string = ' '.join(summ_list)

    # return new text
    new_string =  simple_gen_str(input_string = summ_string, 
              lens = length, 
              temp = T, 
              model_choice = model_dict[old_class])  

    # predicts probability of class of new string
    new_class, new_prob = predict_class(new_string)

    # computes cosine similarity
    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform([input_text,new_string])

    #remove multiple \n
#    new_string = new_string.replace('\n\n\n\n', '')
    
    return remove_end_punct(new_string), [old_class, old_prob], [new_class, new_prob], (tfidf * tfidf.T).A, summ_list                                   


################################
#     __________________________
#    |                         | 
#    |  Applications for Site  |
#    |_________________________|
#    
################################



# here's the homepage
#@application.route('/', methods=['POST', 'GET'])
#def homepage():
#    
#    if request.method == 'POST':
#        
#        gpu_status = GPU_avail()
#        
#        return render_template("index.html", gpu_status = gpu_status)
#        
#    return render_template("index.html")


# here's the homepage
@application.route('/')
def homepage():
        
    return render_template("index.html")

# checks gpu availability
@application.route('/gpu_avail')
def gpu_avail():
    gpu_status = GPU_avail()
    return jsonify(result = gpu_status)

# dictionary for 345M hparam model
model_dict_345 = {'Science': 'atlantic_science_345', 
              'Technology': 'atlantic_technology_345',
              'Business': 'atlantic_business_345',
              'Ideas/Opinion': 'atlantic_ideas_345',
              'Education': 'atlantic_education_345',
              'International': 'atlantic_international_345',
              'Politics': 'atlantic_politics_345',
              'Health': 'atlantic_health_345',
              'Short Story': 'all_short_stories_345',
              'Entertainment': 'atlantic_entertainment_345',
              'Gutenberg':'gutenberg_345'}

# dictionary for 117M hparam model
model_dict_117 = {'Science': 'atlantic_science', 
              'Technology': 'atlantic_technology',
              'Business': 'atlantic_business',
              'Ideas/Opinion': 'atlantic_ideas',
              'Education': 'atlantic_education',
              'International': 'atlantic_international',
              'Politics': 'atlantic_politics',
              'Health': 'atlantic_health',
              'Short Story': 'all_short_stories',
              'Entertainment': 'atlantic_entertainment',
              'Gutenberg':'gutenberg'}



@application.route('/output_gen')
def text_output_gen():
    

    entry_result = request.args.get('user_text')
    temp_val = float(request.args.get('temperature'))
    topic = request.args.get('topic')
    length = int(request.args.get('length'))
    model_type = request.args.get('model_type')
    
    # change to 345 model if 345 is in name
    if '345' in model_type:
        model_dict = model_dict_345
        
    else:
        model_dict = model_dict_117
    
    ## we begin with \n\n\n\n to indicate that we are going to the start of a text
    output_result = simple_gen_str(input_string =entry_result,
                                 lens = length, 
                                 temp = temp_val, 
                                 model_choice = model_dict[topic] )

    # remove end without punctuation
    output_result = remove_end_punct(output_result)

#    form = cgi.FieldStorage()
#    with open ('output_text.txt','w') as fileOutput:
#     #fileOutput.write(form.getValue('user_text'))
#     fileOutput.write(output_result)

    # removes the spontaneous line breaks learned from the corpus
    if topic == 'Gutenberg':
        output_result = output_result.replace('\n\n','~').replace('\n',' ').replace('~','\n\n')

    return render_template("output_gen.html",
                           model_type = model_type,
                           output_text= output_result,
                           entered_text = entry_result, 
                           temp = temp_val,
                           tops = topic,
                           lengs = length,
    )


@application.route('/output_class_gen')
def text_output_class_gen():
    

    entry_result = request.args.get('user_text')
    temp_val = float(request.args.get('temperature'))
    length = int(request.args.get('length'))
    model_type = request.args.get('model_type')
    
    # change to 345 model if 345 is in name
    if '345' in model_type:
        model_dict = model_dict_345
        
    else:
        model_dict = model_dict_117
        
    ## we begin with \n\n\n\n to indicate that we are going to the start of a text
    output_result, old_class_probs, new_class_probs  = class_gen_str(input_text =entry_result,
                                 length = length, 
                                 T = temp_val,
                                 model_dict = model_dict)

    # remove end without punctuation
    output_result = remove_end_punct(output_result)

#    form = cgi.FieldStorage()
#    with open ('output_text.txt','w') as fileOutput:
#     #fileOutput.write(form.getValue('user_text'))
#     fileOutput.write(output_result)

    return render_template("output_class_gen.html",
                           model_type = model_type,
                           old_class = old_class_probs[0],
                           old_prob = round(old_class_probs[1]*100, 1),
                           new_class = new_class_probs[0],
                           new_prob = round(new_class_probs[1]*100, 1),
                           output_class_gen_text= output_result,
                           entered_text = entry_result, 
                           temp = temp_val,
                           lengs = length,
    )

@application.route('/output_class_ex_gen')
def text_output_class_extract_gen():
    
    entry_result = request.args.get('user_text')
    temp_val = float(request.args.get('temperature'))
    length = int(request.args.get('length'))
    model_type = request.args.get('model_type')
    
    if len(sent_tokenize(entry_result)) < 3 :
        return render_template("output_class_ex_gen_error.html")
    
    else: 
        # change to 345 model if 345 is in name
        if '345' in model_type:
            model_dict = model_dict_345

        else:
            model_dict = model_dict_117

        ## we begin with \n\n\n\n to indicate that we are going to the start of a text
        output_result, old_class_probs, new_class_probs, similarity, sum_string  = class_extract_gen_str(input_text = entry_result,
                                     length = length, 
                                     T = temp_val,
                                     model_dict = model_dict)

        # remove end without punctuation
        output_result = remove_end_punct(output_result)

    #    form = cgi.FieldStorage()
    #    with open ('output_text.txt','w') as fileOutput:
    #     #fileOutput.write(form.getValue('user_text'))
    #     fileOutput.write(output_result)

        return render_template("output_class_ex_gen.html",
                               output_class_ex_gen_text= output_result,
                               model_type = model_type,
                               summary1 = sum_string[0],
                               summary2 = sum_string[1],
                               summary3 = sum_string[2],
                               old_class = old_class_probs[0],
                               old_prob = round(old_class_probs[1]*100, 3),
                               new_class = new_class_probs[0],
                               new_prob = round(old_class_probs[1]*100, 3),
                               text_sim = similarity[0][1],
                               entered_text = entry_result, 
                               temp = temp_val,
                               lengs = length,
        )

#@application.route('/contact', methods=['GET', 'POST'])
#def contact():
#    form = ContactForm()
#
#    if request.method == 'POST':
#        return 'Form posted.'
#
#    elif request.method == 'GET':
#        return render_template('contact.html', form=form) 


@application.route('/slides')
def slides():
    return render_template("slides.html")

@application.route('/usage')
def usage():
    return render_template("usage_tips.html")
    
@application.route('/about')
def about():
    return render_template("about.html")

@application.route('/corpora')
def corpora():
    return render_template("corpora.html")

@application.route('/ethics')
def ethics():
    return render_template("ethics.html")

@application.route('/index')
def index():
    return render_template("index.html")
