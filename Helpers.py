import warnings
warnings.filterwarnings("ignore")
import pickle as pk
import pandas as pd
import wordcloud as wc
import gensim as gs
import gensim.corpora as cp
# import pyLDAvis.gensim
import pyLDAvis
from pyLDAvis import gensim_models
import numpy as np
import  matplotlib.pyplot as plt


# TOPIC MODELING
#######################################################################################################
def word_cloud(df):
    all_words = ''
    for text in df['text']:
        all_words += ','.join(list(text))

    print('Total words {}'.format(len(all_words)))

    wordcloud = wc.WordCloud(background_color="white", max_words=100000, contour_width=6, scale=10,
                             contour_color='steelblue')
    wordcloud.generate(all_words)
    return wordcloud.to_image()


def lda(df, n_topics=5, lda_str='all'):
    all_words = []
    for text in df['text']:
        all_words.append(text)

    # Create dictionary and corpus
    word2num = cp.Dictionary(all_words)
    texts = all_words

    # Get term frequency
    corpus = [word2num.doc2bow(text) for text in texts]

    lda_model = gs.models.LdaMulticore(corpus=corpus, id2word=word2num, num_topics=n_topics)
    doc_lda = lda_model[corpus]

    print('\nTopics')
    print(lda_model.print_topics())

    print('\nScores')
    for i in range(0, len(corpus), 500):
        for index, score in sorted(lda_model[corpus[i]], key=lambda tup: -1 * tup[1]):
            print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

    LDAvis_prepared = pyLDAvis.enable_notebook()
    pyLDAvis.save_html(LDAvis_prepared,'./html/{}_lda_n{}.html'.format(lda_str, n_topics))
#######################################################################################################

