import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

import os
from datetime import datetime
from scipy.stats import kurtosis, skew
import itertools
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import randint
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import random
import re

from dateutil import parser
import torch
from transformers import pipeline

pd.options.display.max_columns = 999
import warnings
warnings.filterwarnings('ignore')
import gc
import re
from wordcloud import WordCloud, STOPWORDS
import ipywidgets as widgets
from os.path import exists

stopwords = set(STOPWORDS)

# finbert_esg = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)

# FinBERT is a pre-trained NLP model to analyze sentiment of financial text. 
# It is built by further training the BERT language model in the finance domain, 
# using a large financial corpus and thereby fine-tuning it for financial sentiment classification. 
# Financial PhraseBank by Malo et al. (2014) is used for fine-tuning. 
# For more details, please see the paper FinBERT: Financial Sentiment Analysis with Pre-trained Language Models and our related blog post on Medium.
# https://huggingface.co/ProsusAI/finbert

finbert_sentiment = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer_sentiment = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
sentiment_pipeline = pipeline("text-classification", model=finbert_sentiment, tokenizer=tokenizer_sentiment)

finbert_esg = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)
tokenizer_esg = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
esg_label_pip = pipeline("text-classification", model=finbert_esg, tokenizer=tokenizer_esg)

def parse_dt(s):
    try:
        return(parser.parse(str(s)))
    except:
        return(np.nan)
    


# cols_expect = ['Environmental','Social','Governance']

def get_esg_label_transcript(tr): #get esg labels for every sentence
    sent_label_scores = []

    for sent in sent_tokenize(tr):
        all_esg_labels = esg_label_pip(sent)
        non_none_labels = [x for x in all_esg_labels if x['label']!='None']
        if(len(non_none_labels)>0):
            sent_label_scores.append([non_none_labels[0]['label'],non_none_labels[0]['score'],sent])
    df = pd.DataFrame(sent_label_scores, columns=['esg_label', 'label_score', 'sent'])
    return(df)

def create_sentiment_output(all_labels): #transformation
    non_none_labels = [x for x in all_labels if x['label']!='None']
    if(len(non_none_labels)>0):
        label = non_none_labels[0]['label']
        score = non_none_labels[0]['score']
        sentiment = 0
        if(label=='Positive'):
            return(1*score)
        elif(label=='Negative'):
            return(-1*score)
        else:
            return 0
    else:
        return 0



def get_wordcloud(wordcloud_data_statements):
    comment_words = ''
    for val in wordcloud_data_statements:
        val = str(val)
        tokens = val.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens)+" "
    
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
 
    # plot the WordCloud image                      
    plt.figure(figsize = (4, 4), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()
    return
