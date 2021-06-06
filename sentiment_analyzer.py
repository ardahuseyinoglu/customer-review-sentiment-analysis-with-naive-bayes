# IMPORT LIBRARIES

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# configure
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

# sklearn libraries
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# word cloud
from wordcloud import WordCloud

# manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2 
import os
from tqdm import tqdm 
from zipfile import ZipFile
from PIL import Image
from textblob import TextBlob
import string
import re

#------------------------------------------------------------------------------------------------

# TXT TO DATAFRAME

topic_categories = []
sentiments = []
identifiers = []
reviews = []

with open('all_sentiment_shuffled.txt', encoding='utf8') as file:
    for line in file:
        splitted_line = line.split()
        topic_categories.append(splitted_line[0])
        sentiments.append(splitted_line[1])
        identifiers.append(splitted_line[2])
        reviews.append(' '.join(splitted_line[3:]))

data = pd.DataFrame(data = {'category':topic_categories, 'sentiment':sentiments, 'identifier':identifiers, 'review':reviews})


# remove punctuations
# third argument of maketrans method allows to list all of the characters to remove during the translation process.
table = str.maketrans('', '', string.punctuation)
data['cleared_review'] = data.review.str.translate(table)

#remove digits
data['cleared_review'] = data['cleared_review'].apply(lambda x: re.sub('\w*\d\w*','', x))

# additional columns for statistical analysis
data['length_of_review'] = data['review'].apply(len)
data['number_of_tokens_in_review'] = data['cleared_review'].apply(lambda x: len(str(x).split()))

# gives sentiment polarity score in range[-1,1]. -1 means negative sentiment, 1 means positive sentiment
data['polarity'] = data['review'].map(lambda text: TextBlob(text).sentiment.polarity)

print(data.head())

#-------------------

# SPLITTING DATA
# since data is given as shuffled, only thing we do is taking first 80% of teh data for training and the rest for testing
splitting_index = int(0.8 * len(data))
train_data, test_data = data[:splitting_index], data[splitting_index:]


#-----------------------------------------------------------------------------------------------------------------


# EDA

# Number of token distribution for each topic category
sns.catplot(x="category", y="number_of_tokens_in_review", data=train_data, hue="sentiment",palette="Set1")


# Number of token distribution for each sentiment
sns.catplot(x="sentiment", y="number_of_tokens_in_review", data=train_data, jitter=False, palette="Set1")


# Number of review in each topic category
sns.catplot(x="category", kind="count", palette="Set3", data=train_data)


# Number of review in each sentiment
sns.catplot(x="sentiment", kind="count", palette="Set1", data=train_data)


# Reviews have highest polarity scores (highly positive sentiment)
high_pos_sentiment = train_data.loc[train_data.polarity.sort_values(ascending = False).head(10).index, 'review'].values
for i, review in enumerate(high_pos_sentiment):
    print(i, ' ', review, '\n')


# Reviews have lowest polarity scores (highly negative sentiment)
high_neg_sentiment = train_data.loc[train_data.polarity.sort_values().head(10).index, 'review'].values
for i, review in enumerate(high_neg_sentiment):
    print(i, ' ', review, '\n')


# Neutral reviews (neutral sentiment polarity = 0)
neutral_sentiment = train_data.loc[(train_data.polarity == 0).head(10).index, 'review'].values
for i, review in enumerate(neutral_sentiment):
    print(i, ' ', review, '\n')


# The distribution of review sentiment polarity score
sns.distplot(train_data.polarity)


# The distribution of  text lengths of reviews
sns.distplot(train_data.length_of_review)


# The distribution of number of tokens (words) of reviews
sns.distplot(train_data.number_of_tokens_in_review)


#---------------------------------------------------------------------------------------------------------------------------

# MODEL

def show_dict_content(freq_dict, n_item):
    for label in freq_dict.keys():
        print("-Content of WordFreqDict for class of " + "\'" + label + "\':")
        print("  Prior probability: ", freq_dict[label][0])
        
        print("\n\tFrequency Dictionary - with stopwords:")
        print("\tDictionary Length:", len(freq_dict[label][1]), '\n')
        for key in list(freq_dict[label][1])[:n_item]:
            print('\t', key, ': ', freq_dict[label][1][key])

        print("\n\tFrequency Dictionary - without stopwords:")
        print("\tDictionary Length:", len(freq_dict[label][2]), '\n')
        for key in list(freq_dict[label][2])[:n_item]:
            print('\t', key, ': ', freq_dict[label][2][key])
        
        print("\n")


        
def calculate_freqs(review_series, n_gram):
    # create a dictionary which includes unique words and their freqs for given_series
    freq_dict = {}
    for review in review_series:
        n_gram_list = splitted_review_to_ngram_list(review, n_gram)
        for n_gram_word in n_gram_list:
            if n_gram_word in freq_dict:
                freq_dict[n_gram_word] += 1
            else:
                freq_dict[n_gram_word] = 1
    
    # create descending sorted dictionary according to word freqs
    freq_dict = dict(sorted(freq_dict.items(), key = lambda item: item[1], reverse=True))
    
    return freq_dict



def splitted_review_to_ngram_list(splitted_review, n_gram):
    
    lis = []
    if len(splitted_review) < n_gram:
        return lis

    
    for word_index in range(len(splitted_review)):
        if (word_index == len(splitted_review) - (n_gram-1)) and (n_gram != 1):
            break

        concat_adj_words = ""
        for i in range(n_gram):
            if i == n_gram-1:
                concat_adj_words += splitted_review[word_index + i]
            else:
                concat_adj_words += splitted_review[word_index + i] + ' '

        lis.append(concat_adj_words)
    
    return lis



def remove_stopwords_in_array(splitted_review):
    cleared_splitted_review = []
    
    for word in splitted_review:
        if word not in list(ENGLISH_STOP_WORDS):
            cleared_splitted_review.append(word)
            
    return cleared_splitted_review



def create_word_freq_dict(data, review_column_name, feature_column_name, n_gram):
    
    labels_freq_dicts = {}
    
    # get unique labels for given column
    label_names = list(data[feature_column_name].unique())
    
    for label_name in label_names:
        
        sorted_freq_dict = {}
        sorted_freq_dict_no_stopwords = {}
   
        # each element in the series is the array of splitted words for corresponding review.
        splitted_review_series = data[data[feature_column_name] == label_name][review_column_name].str.split()

        #each element in the series is the array of splitted words for corresponding review withoout stop words.
        splitted_review_no_stopword_series = splitted_review_series.apply(remove_stopwords_in_array)

        sorted_freq_dict = calculate_freqs(splitted_review_series, n_gram)
        sorted_freq_dict_no_stopwords = calculate_freqs(splitted_review_no_stopword_series, n_gram) 
        
        prior_prob = len(data[data[feature_column_name] == label_name]) / len(data)
        
        labels_freq_dicts[label_name] = [prior_prob, sorted_freq_dict, sorted_freq_dict_no_stopwords]
        
    return labels_freq_dicts



def show_wordcloud(a_dict, plt_title):
    wordcloud = WordCloud(width=900,
                          height=500, 
                          max_words=100,
                          relative_scaling=1,
                          normalize_plurals=False, 
                          background_color="white").generate_from_frequencies(a_dict)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(plt_title, fontsize=15, fontname='Georgia',  y=-0.22)
    plt.axis("off")
    plt.show()


    
    
sentiment_labels_freq_dicts_uni = create_word_freq_dict(train_data, 'cleared_review', 'sentiment', n_gram=1)
category_labels_freq_dicts_uni = create_word_freq_dict(train_data, 'cleared_review', 'category', n_gram=1)

sentiment_labels_freq_dicts_bi = create_word_freq_dict(train_data, 'cleared_review', 'sentiment', n_gram=2)
category_labels_freq_dicts_bi = create_word_freq_dict(train_data, 'cleared_review', 'category', n_gram=2)

sentiment_labels_freq_dicts_tri = create_word_freq_dict(train_data, 'cleared_review', 'sentiment', n_gram=3)

# Show barplots for frequencies of words vs classes
my_df = pd.DataFrame(sentiment_labels_freq_dicts_bi['pos'][2].items()).head(20)
ax = sns.barplot(x=1, y=0, data=my_df, palette="mako")
ax.set(xlabel = 'Frequency', ylabel='Word', title='Top20 Frequent Word in Positive Sentimented Reviews - stopwords not included\n')


print("WordFreqDicts for \'sentiment\' column - unigram:\n")
show_dict_content(sentiment_labels_freq_dicts_uni,25)

print("WordFreqDicts for \'sentiment\' column - bigram:\n")
show_dict_content(sentiment_labels_freq_dicts_bi,25)

# display wordclouds for each feature and their classes
show_wordcloud(sentiment_labels_freq_dicts_uni['neg'][2], 'Negative Sentiment Reviews')
show_wordcloud(sentiment_labels_freq_dicts_uni['pos'][2], 'Positive Sentiment Reviews')
show_wordcloud(category_labels_freq_dicts_uni['music'][2], 'Music Category Reviews')
show_wordcloud(category_labels_freq_dicts_uni['books'][2], 'Books Category Reviews')
show_wordcloud(category_labels_freq_dicts_uni['dvd'][2], 'DVD Category Reviews')
show_wordcloud(category_labels_freq_dicts_uni['camera'][2], 'Camera Category Reviews')
show_wordcloud(category_labels_freq_dicts_uni['health'][2], 'Health Category Reviews')
show_wordcloud(category_labels_freq_dicts_uni['software'][2], 'Software Category Reviews')


#--------------------------------------------------------------------------------------------------------------------------------------


# TESTING

# stop_word: if 1 --> stopwords are included, if 2 --> no stopword.
def create_model_and_test(test_data, train_data, review_column_name, feature_column_name, n_gram, train_stop_word, test_stop_word):
    
    print("Corresponding frequency dictionaries are being created...")
    freq_dicts = create_word_freq_dict(train_data, review_column_name, feature_column_name, n_gram)
    print("Done.")
    
    freq_dicts_for_V = create_word_freq_dict(train_data, review_column_name, feature_column_name, n_gram=1)
    V = get_V_size(freq_dicts_for_V, train_stop_word)
    
    labels = list(freq_dicts.keys())
    gt_values = test_data[feature_column_name].replace(to_replace=labels, value=np.arange(len(labels)))
    splitted_test_reviews =  split_test_data(test_data, review_column_name, n_gram, test_stop_word)
    prob_scores_for_reviews = np.empty([len(splitted_test_reviews), len(labels)])
    
    print("Testing's started...")
    for i, splitted_review in enumerate(tqdm(splitted_test_reviews, position=0, leave=True)):
        
        for j, label in enumerate(labels):

            total_freq_in_class = sum(freq_dicts[label][train_stop_word].values())
            word_likelihoods = []

            for word in splitted_review:

                word_freq_in_class  = freq_dicts[label][train_stop_word].get(word, 0)
                likelihood_of_word = ((word_freq_in_class + 1) / (total_freq_in_class + V)) 
                word_likelihoods.append(likelihood_of_word)

            review_label_prob = np.sum(np.log(np.array(word_likelihoods))) + np.log(freq_dicts[label][0])
            prob_scores_for_reviews[i][j] = review_label_prob

    preds = np.argmax(prob_scores_for_reviews, axis=1)
    return preds, gt_values, labels



def split_test_data(test_data, review_column_name, n_gram, stop_word):
    
    if stop_word == 1:
        test_splitted_reviews = test_data[review_column_name].str.split()
    elif stop_word == 2:   
        test_splitted_reviews = test_data[review_column_name].str.split().apply(remove_stopwords_in_array)
    
    if n_gram == 1:
        return test_splitted_reviews
    
    else:
        test_splitted_reviews_n = []
        for splitted_rev in test_splitted_reviews:
            test_splitted_reviews_n.append(splitted_review_to_ngram_list(splitted_rev, n_gram))
        
        return test_splitted_reviews_n



# dict_no --> 1: with stopwords, 2: without stopwords
def get_V_size(freq_dicts,dict_no):
    duplicated_all_vocab = []
    for label in freq_dicts.keys():
        duplicated_all_vocab.extend(list(freq_dicts[label][dict_no].keys())) 
    v_size = len(set(duplicated_all_vocab))
    return v_size



def get_accuracy(preds, y):
    accuracy = np.sum((preds - y) == 0) / len(preds)
    return accuracy*100


# Sentiment Classification
# stop_word: if 1 --> stopwords are included, if 2 --> no stopword.
preds, gt_values, labels = create_model_and_test(test_data, 
                                                 train_data, 
                                                 review_column_name='review', 
                                                 feature_column_name='sentiment', 
                                                 n_gram=2, 
                                                 train_stop_word=1, 
                                                 test_stop_word=1)

print(get_accuracy(preds, gt_values))

# confusion matrix
sns.heatmap(confusion_matrix(gt_values, preds), 
            annot=True, 
            fmt="d", 
            cmap="YlGnBu",
            xticklabels=labels, 
            yticklabels=labels)

# classification report
print(classification_report(gt_values, preds, target_names=labels))


# Misclassified samples
misclassified_reviews_index = np.where(((preds - gt_values) != 0).values == True)[0]
print(misclassified_reviews_index)

def print_misclasified_sample(n):
    print("Ground truth label is:", labels[gt_values.values[n]], ", predicted as:", labels[preds[n]], "\n")
    print("The review is:\n", test_data.review.values[n])

print_misclasified_sample(1824)


# with tf-idf weights:

frq_dic = create_word_freq_dict(train_data, 'review', 'sentiment', n_gram=1)

idfs_pos = {}
total_pos_docs = len(train_data[train_data['sentiment'] == 'pos'])
for word in tqdm(frq_dic['pos'][1].keys(), position=0, leave=True):
    n_docs_word_pass = len(train_data[train_data['review'].str.contains(word, regex=False)])
    try:
        idfs_pos[word] = np.log(total_pos_docs/n_docs_word_pass)
    except:
        print(word)

        
idfs_neg = {}
total_neg_docs = len(train_data[train_data['sentiment'] == 'neg'])
for word in tqdm(frq_dic['neg'][1].keys(), position=0, leave=True):
    n_docs_word_pass = len(train_data[train_data['review'].str.contains(word, regex=False)])
    try:
        idfs_neg[word] = np.log(total_neg_docs/n_docs_word_pass)
    except:
        print(word)

        
tf_idf_neg = pd.concat([pd.Series(idfs_neg), pd.Series(frq_dic['neg'][1])], axis=1)
td_idf_weights_neg = abs(tf_idf_neg.loc[:,0] * tf_idf_neg.loc[:,1])

tf_idf_pos = pd.concat([pd.Series(idfs_pos), pd.Series(frq_dic['pos'][1])], axis=1)
td_idf_weights_pos = abs(tf_idf_pos.loc[:,0] * tf_idf_pos.loc[:,1])

tf_idf_weights_dict = {'neg': td_idf_weights_neg, 'pos': td_idf_weights_pos}



# stop_word: if 1 --> stopwords are included, if 2 --> no stopword.
def create_model_and_test_tf_idf(test_data, train_data, review_column_name, feature_column_name, n_gram, train_stop_word, test_stop_word):
    
    print("Corresponding frequency dictionaries are being created...")
    freq_dicts = create_word_freq_dict(train_data, review_column_name, feature_column_name, n_gram)
    print("Done.")
    
    freq_dicts_for_V = create_word_freq_dict(train_data, review_column_name, feature_column_name, n_gram=1)
    V = get_V_size(freq_dicts_for_V, train_stop_word)
    
    labels = list(freq_dicts.keys())
    gt_values = test_data[feature_column_name].replace(to_replace=labels, value=np.arange(len(labels)))
    splitted_test_reviews =  split_test_data(test_data, review_column_name, n_gram, test_stop_word)
    prob_scores_for_reviews = np.empty([len(splitted_test_reviews), len(labels)])
    
    print("Testing's started...")
    for i, splitted_review in enumerate(tqdm(splitted_test_reviews, position=0, leave=True)):
        
        for j, label in enumerate(labels):

            total_freq_in_class = sum(freq_dicts[label][train_stop_word].values())
            word_likelihoods = []

            for word in splitted_review:
                likelihood_of_word = ((tf_idf_weights_dict[label].get(word, 0) + 1) / (tf_idf_weights_dict[label].sum() + V))     
                word_likelihoods.append(likelihood_of_word)

            review_label_prob = np.sum(np.log(np.array(word_likelihoods))) + np.log(freq_dicts[label][0])
            prob_scores_for_reviews[i][j] = review_label_prob

    preds = np.argmax(prob_scores_for_reviews, axis=1)
    return preds, gt_values, labels


# get prediction for tf-idf model
# stop_word: if 1 --> stopwords are included, if 2 --> no stopword.
preds, gt_values, labels = create_model_and_test_tf_idf(test_data, 
                                                 train_data, 
                                                 review_column_name='review', 
                                                 feature_column_name='sentiment', 
                                                 n_gram=1, 
                                                 train_stop_word=1, 
                                                 test_stop_word=1)

get_accuracy(preds, gt_values)


# Category Classification

# stop_word: if 1 --> stopwords are included, if 2 --> no stopword.
preds, gt_values, labels = create_model_and_test(test_data, 
                                                 train_data, 
                                                 review_column_name='review', 
                                                 feature_column_name='category', 
                                                 n_gram=1, 
                                                 train_stop_word=1, 
                                                 test_stop_word=1)


get_accuracy(preds, gt_values)

# confusion matrix
sns.heatmap(confusion_matrix(gt_values, preds), 
            annot=True, 
            fmt="d", 
            cmap="YlGnBu",
            xticklabels=labels, 
            yticklabels=labels)


# classification report
print(classification_report(gt_values, preds, target_names=labels))


# Modul Analysis

# Stopwords are not included, Punctuations are removed +  Words of intersection of two classes + Unigram
intersection_words = set(sentiment_labels_freq_dicts_uni['neg'][2].keys()) & set(sentiment_labels_freq_dicts_uni['pos'][2].keys())

neg_freqs_interseciton = {}
for key in list(intersection_words):
    neg_freqs_interseciton[key] = sentiment_labels_freq_dicts_uni['neg'][2].get(key)
neg_freqs_interseciton = dict(sorted(neg_freqs_interseciton.items(), key = lambda item: item[1], reverse=True))   


pos_freqs_interseciton = {}
for key in list(intersection_words):
    pos_freqs_interseciton[key] = sentiment_labels_freq_dicts_uni['pos'][2].get(key)
pos_freqs_interseciton = dict(sorted(pos_freqs_interseciton.items(), key = lambda item: item[1], reverse=True))    


freq_diff_dict_pos = {}
for key in pos_freqs_interseciton.keys():
    freq_diff = pos_freqs_interseciton.get(key) - neg_freqs_interseciton.get(key)
    freq_diff_dict_pos[key] = freq_diff
freq_diff_dict_pos_presence_uni = dict(sorted(freq_diff_dict_pos.items(), key = lambda item: item[1], reverse=True)) 
freq_diff_dict_pos_absence_uni = dict(sorted(freq_diff_dict_pos.items(), key = lambda item: item[1]))


freq_diff_dict_neg = {}
for key in neg_freqs_interseciton.keys():
    freq_diff = neg_freqs_interseciton.get(key) - pos_freqs_interseciton.get(key)
    freq_diff_dict_neg[key] = freq_diff
freq_diff_dict_neg_presence_uni = dict(sorted(freq_diff_dict_neg.items(), key = lambda item: item[1], reverse=True))  
freq_diff_dict_neg_absence_uni = dict(sorted(freq_diff_dict_neg.items(), key = lambda item: item[1])) 


# Stopwords are not included, Punctuations are removed +  Words of intersection of two classes + Bigram

intersection_words = set(sentiment_labels_freq_dicts_bi['neg'][2].keys()) & set(sentiment_labels_freq_dicts_bi['pos'][2].keys())

neg_freqs_interseciton = {}
for key in list(intersection_words):
    neg_freqs_interseciton[key] = sentiment_labels_freq_dicts_bi['neg'][2].get(key)
neg_freqs_interseciton = dict(sorted(neg_freqs_interseciton.items(), key = lambda item: item[1], reverse=True))   


pos_freqs_interseciton = {}
for key in list(intersection_words):
    pos_freqs_interseciton[key] = sentiment_labels_freq_dicts_bi['pos'][2].get(key)
pos_freqs_interseciton = dict(sorted(pos_freqs_interseciton.items(), key = lambda item: item[1], reverse=True))


freq_diff_dict_pos = {}
for key in pos_freqs_interseciton.keys():
    freq_diff = pos_freqs_interseciton.get(key) - neg_freqs_interseciton.get(key)
    freq_diff_dict_pos[key] = freq_diff
freq_diff_dict_pos_presence_bi = dict(sorted(freq_diff_dict_pos.items(), key = lambda item: item[1], reverse=True)) 
freq_diff_dict_pos_absence_bi = dict(sorted(freq_diff_dict_pos.items(), key = lambda item: item[1]))


freq_diff_dict_neg = {}
for key in neg_freqs_interseciton.keys():
    freq_diff = neg_freqs_interseciton.get(key) - pos_freqs_interseciton.get(key)
    freq_diff_dict_neg[key] = freq_diff
freq_diff_dict_neg_presence_bi = dict(sorted(freq_diff_dict_neg.items(), key = lambda item: item[1], reverse=True))  
freq_diff_dict_neg_absence_bi = dict(sorted(freq_diff_dict_neg.items(), key = lambda item: item[1])) 


my_df = pd.DataFrame(freq_diff_dict_pos_presence.items()).head(20)
ax = sns.barplot(x=1, y=0, data=my_df, palette="rocket")
ax.set(xlabel = 'Frequency', ylabel='Word', title='Top20 words whose presence most strongly predicts that the review is positive\n or \n Top20 words whose absence most strongly predicts that the review is negative\n')


# Stopwords are not included, Punctuations are removed +  Class Exceptions + Unigram: 


only_pos = set(sentiment_labels_freq_dicts_uni['pos'][2].keys()) - set(sentiment_labels_freq_dicts_uni['neg'][2].keys())
only_pos_freqs = {}
for key in list(only_pos):
    only_pos_freqs[key] = sentiment_labels_freq_dicts_uni['pos'][2].get(key)
only_pos_freqs = dict(sorted(only_pos_freqs.items(), key = lambda item: item[1], reverse=True)) 


only_neg = set(sentiment_labels_freq_dicts_uni['neg'][2].keys()) - set(sentiment_labels_freq_dicts_uni['pos'][2].keys())
only_neg_freqs = {}
for key in list(only_neg):
    only_neg_freqs[key] = sentiment_labels_freq_dicts_uni['neg'][2].get(key)
only_neg_freqs = dict(sorted(only_neg_freqs.items(), key = lambda item: item[1], reverse=True)) 



#------------------------------------------------------------------------------------------------------------------------
# for kaggle competition
indices = []
reviews = []

with open('data_wlabel.csv', encoding='utf8') as file:
    for line in file:
        splitted_line = line.split(",", 1)
        indices.append(splitted_line[0])
        reviews.append(splitted_line[1])
        
test_data = pd.DataFrame(data = {'indices':indices, 'review':reviews})
train_data = data

# stop_word: if 1 --> stopwords are included, if 2 --> no stopword.
def test_model(test_data, train_data, review_column_name, feature_column_name, n_gram, train_stop_word, test_stop_word):
    
    print("Corresponding frequency dictionaries are being created...")
    freq_dicts = create_word_freq_dict(train_data, review_column_name, feature_column_name, n_gram)
    print("Done.")
    
    freq_dicts_for_V = create_word_freq_dict(train_data, review_column_name, feature_column_name, n_gram=1)
    V = get_V_size(freq_dicts_for_V, train_stop_word)
    
    labels = list(freq_dicts.keys())
    splitted_test_reviews =  split_test_data(test_data, review_column_name, n_gram, test_stop_word)
    prob_scores_for_reviews = np.empty([len(splitted_test_reviews), len(labels)])
    
    print("Testing's started...")
    for i, splitted_review in enumerate(tqdm(splitted_test_reviews, position=0, leave=True)):
        
        for j, label in enumerate(labels):

            total_freq_in_class = sum(freq_dicts[label][train_stop_word].values())
            word_likelihoods = []

            for word in splitted_review:

                word_freq_in_class  = freq_dicts[label][train_stop_word].get(word, 0)
                likelihood_of_word = ((word_freq_in_class + 1) / (total_freq_in_class + V)) 
                word_likelihoods.append(likelihood_of_word)

            review_label_prob = np.sum(np.log(np.array(word_likelihoods))) + np.log(freq_dicts[label][0])
            prob_scores_for_reviews[i][j] = review_label_prob

    preds = np.argmax(prob_scores_for_reviews, axis=1)
    return preds

# stop_word: if 1 --> stopwords are included, if 2 --> no stopword.
preds= test_model(test_data, 
                train_data, 
                review_column_name='review', 
                feature_column_name='sentiment', 
                n_gram=2, 
                train_stop_word=1, 
                test_stop_word=1)

submission_df = pd.DataFrame({'Id':np.arange(0,2382), 'Category':preds})
submission_df = submission_df.astype({"Id": 'int64', 'Category': str})
submission_df['Category'] = submission_df['Category'].str.replace('0','neg')
submission_df['Category'] = submission_df['Category'].str.replace('1','pos')
submission_df.to_csv('submission_ah.csv', index=False)






