#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import re as re
import nltk
import collections
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[3]:


df= pd.read_csv('/Users/bingyandu/Downloads/kickstartergame.csv', sep=',',dtype=str).apply(lambda x: x.astype(str).str.lower())
#df.head()


# In[4]:


corpus = df['Description']
#corpus


# In[5]:


# preprocessing 
def preprocess(corpus):
    clean_data = []
    for x in (corpus[:]):
        new_text = re.sub('<.*?>', '', x)   
        new_text = re.sub(r'[^\w\s]', '', new_text) 
        new_text = re.sub(r'\d+','',new_text)         
        if new_text != '':
            clean_data.append(new_text)
    return clean_data

new1=preprocess(corpus)
# print(new1)


# In[6]:


#word tokenization
from nltk.tokenize import word_tokenize


def tokeni(new1):
    s_token = []
    for y in new1:
        new_word = word_tokenize(y)
        if new_word != '':
            s_token.append(new_word)
    return s_token


new2 = tokeni(new1)
# print(new2)
#print(type(new2))
#print(type(new2[3]))


# In[62]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(new2, min_count=5, threshold=100) 
trigram = gensim.models.Phrases(bigram[new2], threshold=100)  


# In[63]:


# Build the bigram and trigram models - conn
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[11]:


# remove stopwords
from nltk.corpus import stopwords

stop = stopwords.words('english')
new3=[]
for sentence in new2:
    new3.append([word for word in sentence if word not in stop])

#print(new2[0])
#print(new3[0])


# In[9]:


#define functions for bigran and trigrams
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# In[12]:


#call bigram
data_words_bigrams = make_bigrams(new3)


# In[13]:


#stemming
from nltk.stem.snowball import SnowballStemmer
snowball = SnowballStemmer(language = 'english')
def stemming(new3):
    new = []
    for sentence in new3:
        stem_words = [snowball.stem(x) for x in sentence]
        new.append(stem_words)
    return new

new4 = stemming(data_words_bigrams)
# print(new3[0])
# print(new4[0])


# In[14]:


# lemming
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatization(new4):
    new = []
    for sentence in new4:
        lem_words = [lemmatizer.lemmatize(x) for x in sentence]
        new.append(lem_words)
    return new

new5 = lemmatization(new4)
# print(new4[0])
# print(new5[0])


# In[15]:


# Create Dictionary
id2word = corpora.Dictionary(new5)


# In[17]:


# Create Corpus
texts = new5


# In[19]:


# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]


# In[21]:


#download mallet
mallet = '/Users/bingyandu/Downloads/mallet-2.0.8/bin/mallet'


# In[28]:


#use gensim wrapper for mallet
ldamallet = gensim.models.wrappers.LdaMallet(mallet, corpus=corpus, num_topics=15, id2word=id2word)


# In[32]:


#Define the function to calcuate the coherence_values 
# Coherence values corresponding to the LDA model with respective number of topics
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):

    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print('Calculating {}-topic model'.format(num_topics))
        model = gensim.models.wrappers.LdaMallet(mallet, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[69]:


print(ldamallet.show_topics(num_topics=1000, formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=new5, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


# In[33]:


limit=35; start=2; step=1;
model_list, coherence_values = compute_coherence_values(dictionary=id2word,
                                                        corpus=corpus,
                                                        texts=new5,
                                                        start=start,
                                                        limit=limit,
                                                        step=step)


# In[34]:


# Show graph of number of topics agains coherence score
x = range(start, limit, step)
plt.figure(figsize=(15, 10))
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
plt.show()


# In[35]:


for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 6))


# In[55]:


import operator
# Select the model and print the topics
index, value = max(enumerate(coherence_values), key=operator.itemgetter(1))
index = 14
optimal_model = model_list[index]
model_topics = optimal_model.show_topics(num_topics=1000, formatted=False)


# In[60]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=15, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Compute Perplexity
print ('Perplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=new5, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print ('Coherence Score: ', coherence_lda)

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.display(vis)


# In[61]:


for topic in sorted(optimal_model.show_topics(num_topics=1000, num_words=10, formatted=False), key=lambda x: x[0]):
    print('Topic {}: {}'.format(topic[0], [item[0] for item in topic[1]]))


# In[57]:



def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)
df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=texts)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']


# In[58]:


sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet


topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = sent_topics_sorteddf_mallet[['Topic_Num', 'Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Percent_Documents']

# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]


# In[59]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    pd.set_option('display.max_colwidth', -1)
    display(df_dominant_topics)


# In[ ]:




