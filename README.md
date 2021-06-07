# Sentiment Analysis with Naive Bayes

In this assignment, we implement a naive bayes classifier using the bag of words(BoW) representation with unigram and bigram options and use it to try determining whether a customer review is positive or negative. We also extend our classification task to predicting cateagories(books, camera, dvd, health, music, or software) of the given review. We will also use different techniques to improve our model performance such that removing stop words and adapting tf-idf weights into naive bayes classification problem.


<p align="center">
  <b>Data Analysis</b>
</p>

<p align="center">
  <img src="/report-images/data.PNG">
</p>

We see that we have balanced traning data, since the training data samples are distributed equally in terms of the number of review for each class of sentiment feature and for each class of category feature.

Since we use BoW model, we want that each sentiment should be represented by dictionaries whose sum of word frequencies are equal and they should include variety of unique words that reflects characteristics of its sentiment. So, the number of words used in reviews are basically more important than the number of reviews when we decide if we use a balanced dataset, since a review can be just a few words or just a few paragraphs long. Accordingly, we observe that the distribution of number of words used in a review are balanced based on their sentiments. In addition, each category (music, books, dvd, camera, health, software) has equally distributed reviews for each sentiment feature.


<p align="center">
  <b>Preview on Sentiments: The distribution of review sentiment polarity score</b>
</p>

<p align="center">
  <img src="/report-images/polarity.png">
</p>

<br><br>
<p align="center">
  <b>Preview on Sentiments: Example sentences according their polarity scores</b>
</p>

<p align="center">
  <img src="/report-images/words.PNG">
</p>

We also wanted to use polarity scores to get an intuition and see examples of highly positive reviews which have high polarity score, highly negative reviews which have low polarity score and neutral reviews which have 0 polarity score. Polarity scores is in range [-1,1]. -1 means highest negative sentiment score and 1 means highest positive sentiment score. You can see the review examples above based on their score. We can clearly see that some keywords dominate the review's polarity score and their sentiment class accordingly. "best, happy, excellent, impressed, perfectly ..." are the words used frequently in highly positive sentiments, while "boring, worst, talentless, terrible, horrible, hated ..." are the words used frequently in highly negative sentiments. When it comes to neutral reviews, they generally includes both words used in negative sentiments and positive sentiments. And they are longer reviews comparison with negative and positive reviews. Since they are longer, they include more words. The more words the more emotions, the more expressions. Thus, they can include characteristics words in both sentimen classes, and so that makes them neutral.

<p align="center">
  <b>WordClouds for Sentiments and Categories - without stopwords</b>
</p>

<p align="center">
  <img src="/report-images/word-cloud.PNG">
</p>

Top frequent keywords and the number of how often they appear both in positive and negative classes (both with stopwords included and not included) :


<p align="center">
  <b>Unigram</b>
</p>

<p align="center">
  <img src="/report-images/unigram.PNG">
</p>


<p align="center">
  <b>Bigram</b>
</p>

<p align="center">
  <img src="/report-images/bigram.PNG">
</p>


<p align="center">
  <b>Accuracy scores for all experiments done</b>
</p>

<p align="center">
  <img src="/report-images/acc-table.PNG">
</p>

We use different settings to get better accuracy on our model. We use BoW with three options: unigram, bigram, trigram. We have two option for review which are given review and cleared review. Cleared review means that we remove punctuations and numbers from each review in dataset. We also try our model by removing stopwords from reviews. In addtion, we try to observe how the accuracy will change, if we remove stopword from the model but not from the test set and vice versa. In the end, we get our best model with 86.32% accuracy via bigram BoW model, non-cleared reviews and stopwords are included.

To implement TF-IDF in terms of the the conditional probabilities used in the Naive Bayes algorithm, we use tf-idf weight of the word, which is obtained by multiplying frequency of word with idf value of the word, instead of frequency of the word in the class and use sum of tf-idf weights of the class instead of total word frequency of the class. IDF values are obtained by getting the log of the total number of documents belonging the class over how many documents include that word. However, we get lower accuracy score (73.3%) than the score we get from the model we use only frequencies.


<p align="center">
  <b>Confusion Matrix and Classification Report for the Best Model</b>
</p>

<p align="center">
  <img src="/report-images/conf-sent.PNG">
</p>

<p align="center">
  <b>Some of the misclassified samples</b>
</p>

<p align="center">
  <img src="/report-images/words2.PNG">
</p>

<p style="font-family:Times New Roman; font-size:18px"> There can be several reasons why the model misclassifies the reviews. We list a following possible reasons and give corresponding examples above:</p> 

<ul style="font-family:Times New Roman; font-size:18px">
  <li> (1) Since we use bigram option for the BoW model, we take successive words when creating the model. There are many verbs with negations (e.g. n't). Although the overall sentiment is positive, those negations may make the model predict the sentiment as negative. </li>
  <li> (2) There are many noun and adjectives,which have characteristics of negative sentiment, in the review; although it's a positive review.</li>
  <li> (3) There is sarcasm in the review and our simple model may not be able to understand this. </li>
  <li> (4) In the review, the user quotes negative comments made by someone else and use negative expressions when criticizing other comments. That may makes the model predict the sentiment as negative, because the model thinks using expressions and words in review belong to user's own opinion.</li>
  <li> (5) While it's a negative comment, there are positive words and expressions in it. In addtion, user actually gives a positive comment for other products in her/his review. </li>
</ul>

If the absence of a word increases the likelihood that the model will predict a review as positive, then the presence of this word also increases the likelihood that the model will predict a review as negative, vice versa.


The words whose presence/absence most strongly predicts that the review is positive/negative (with stopwords are not included):

<p align="center">
  <img src="/report-images/data-a.PNG">
</p>

Lastly, we also try to observe what words only belong to positive sentiment dictionary and only belong to negative sentiment dictionar. However, these words has low frequency and they do not express something positive or positive. So they do not make sense to use them to narrow the dictionaries. Top frequent words for both class as follows:

<ul style="font-family:Times New Roman; font-size:18px">
  <li>For positive sentiments: 'pluto': 60, 'lieutenant': 28, 'powell': 23, 'maizon': 23, 'flaxseed': 22, 'picard': 19, 'jakes': 18, 'photographers': 17, 'snack': 17, 'snapfire': 17</li>
  <li>For negative sentiments: 'seagal': 41, 'nis': 27, 'lisbon': 24, 'blah': 24, 'ripoff': 23, 'the\x1a': 22, 'gnr': 22, 'dennett': 19, 'trite': 18, 'bras': 18</li>
</ul>


<p align="center">
  <b>Confusion Matrix and Classification Report for Category Feature</b>
</p>

<p align="center">
  <img src="/report-images/conf-cat.PNG">
</p>

