import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import errno
import platform
from Helpers import word_cloud, lda, get_sentiment
from sklearn.preprocessing import MaxAbsScaler

st.set_page_config(layout="wide")
st.title('Twitter Complaints Analysis')

# SIDEBAR
#######################################################################################################
st.sidebar.image('./images/uber_logo.png')

st.sidebar.markdown('**By:** Anukriti, Siddharth, Richard, Bogdan')
st.sidebar.markdown('**McGill MMA** - Social Media Analytics')
st.sidebar.markdown('**GitHub:** https://github.com/bogdan-tanasie/SMF')

st.sidebar.header('Controls')
st.sidebar.subheader('All Models')
timeline = st.sidebar.selectbox(
    "Select timeframe for analysis",
    ("All", "Pre-COVID", "Post-COVID")
)

st.sidebar.subheader('Topic & Sentiment Models')
direction = st.sidebar.selectbox(
    "Select data direction",
    ("All", "To Uber", "From Uber")
)
n_topics = st.sidebar.selectbox(
    "Select number of topics",
    ("3", "5", "10")
)
#######################################################################################################


# DATA/VARIABLES
#######################################################################################################
cwd = os.getcwd()
os_type = platform.system()


@st.cache(persist=True)
def load_data():

    df_initial = pd.read_pickle(r'./data/uber_tk.p')
    return df_initial

@st.cache(persist=True)
def load_data_Clique():
    df_comm = pd.read_csv(r'./Communities.csv')
    cliques = pd.read_csv("Some_cliques.csv")
    cliques = cliques.sort_values(by="clique_size",ascending=False)
    source_cliques = pd.read_csv("mentions_source_cliques.csv")
    target_cliques = pd.read_csv("mentions_targ_cliques.csv")
    return cliques, source_cliques, target_cliques, df_comm

@st.cache(persist=True, allow_output_mutation=True)
def filter_data(df, t, d, covid_date):
    filter_str = 'all'
    if t == 'All' and d == 'To Uber':
        filter_str = 'to'
        return (df[((df['target'] == 'Uber_Support') | (df['target'] == 'UberVirgDetroit') | (df['target'] == 'Uber_India') | (df['target'] == 'UberEats') | (df['target'] == 'Uber_Kolkata') | (df['target'] == 'Uber') | (df['target'] == 'UberUKsupport') | (df['target'] == 'Uber_MEX') | (df['target'] == 'Uber_RSA') | (df['target'] == 'UberINSupport'))]
                , filter_str)
    elif t == 'All' and d == 'From Uber':
        filter_str = 'from'
        return (df[((df['source'] == 'Uber_Support') | (df['source'] == 'UberVirgDetroit') | (df['source'] == 'Uber_India') | (df['source'] == 'UberEats') | (df['source'] == 'Uber_Kolkata') | (df['source'] == 'Uber') | (df['source'] == 'UberUKsupport') | (df['source'] == 'Uber_MEX') | (df['source'] == 'Uber_RSA') | (df['source'] == 'UberINSupport'))]
                , filter_str)
    elif t == 'Pre-COVID' and d == 'All':
        filter_str = 'pre'
        return df[df['created_at'] <= covid_date], filter_str
    elif t == 'Post-COVID' and d == 'All':
        filter_str = 'post'
        return df[df['created_at'] > covid_date], filter_str
    elif t == 'Pre-COVID' and d == 'To Uber':
        filter_str = 'pre_to'
        return (df[((df['target'] == 'Uber_Support') | (df['target'] == 'UberVirgDetroit') | (df['target'] == 'Uber_India') | (df['target'] == 'UberEats') | (df['target'] == 'Uber_Kolkata') | (df['target'] == 'Uber') | (df['target'] == 'UberUKsupport') | (df['target'] == 'Uber_MEX') | (df['target'] == 'Uber_RSA') | (df['target'] == 'UberINSupport'))
                   & (df['created_at'] <= covid_date)]
                , filter_str)
    elif t == 'Pre-COVID' and d == 'To Uber':
        filter_str = 'pre_from'
        return (df[((df['source'] == 'Uber_Support') | (df['source'] == 'UberVirgDetroit') | (df['source'] == 'Uber_India') | (df['source'] == 'UberEats') | (df['source'] == 'Uber_Kolkata') | (df['source'] == 'Uber') | (df['source'] == 'UberUKsupport') | (df['source'] == 'Uber_MEX') | (df['source'] == 'Uber_RSA') | (df['source'] == 'UberINSupport'))
                   & (df['created_at'] <= covid_date)]
                , filter_str)
    elif t == 'Post-COVID' and d == 'To Uber':
        filter_str = 'post_to'
        return (df[((df['target'] == 'Uber_Support') | (df['target'] == 'UberVirgDetroit') | (df['target'] == 'Uber_India') | (df['target'] == 'UberEats') | (df['target'] == 'Uber_Kolkata') | (df['target'] == 'Uber') | (df['target'] == 'UberUKsupport') | (df['target'] == 'Uber_MEX') | (df['target'] == 'Uber_RSA') | (df['target'] == 'UberINSupport'))
                   & (df['created_at'] > covid_date)]
                , filter_str)
    elif t == 'Post-COVID' and d == 'From Uber':
        filter_str = 'post_from'
        return (df[((df['source'] == 'Uber_Support') | (df['source'] == 'UberVirgDetroit') | (df['source'] == 'Uber_India') | (df['source'] == 'UberEats') | (df['source'] == 'Uber_Kolkata') | (df['source'] == 'Uber') | (df['source'] == 'UberUKsupport') | (df['source'] == 'Uber_MEX') | (df['source'] == 'Uber_RSA') | (df['source'] == 'UberINSupport'))
                   & (df['created_at'] > covid_date)]
                , filter_str)
    else:
        return df, filter_str


@st.cache(persist=True)
def load_network_results(filter_str):
    if 'post' in filter_str:
        return pd.read_csv(r'./data/post_network_results.csv')
    elif 'pre' in filter_str:
        return pd.read_csv(r'./data/pre_network_results.csv')
    else:
        return pd.read_csv(r'./data/all_network_results.csv')

@st.cache(persist=True)
def load_sentiment_results(filter_str, covid_date):
    complaints_sentiment_df = pd.read_csv(r'./data/complaints_sentiment.csv')
    df = pd.read_csv(r'./data/uber_df.csv')
    df.drop(df.columns[0], axis=1, inplace=True)
    df.drop(columns=['source_id', 'target_id', 'all_data'], inplace=True)
    if 'post' in filter_str:
        return (df[((df['target'] == 'Uber_Support') | (df['target'] == 'UberVirgDetroit') | (
                    df['target'] == 'Uber_India') | (df['target'] == 'UberEats') | (df['target'] == 'Uber_Kolkata') | (
                                df['target'] == 'Uber') | (df['target'] == 'UberUKsupport') | (
                                df['target'] == 'Uber_MEX') | (df['target'] == 'Uber_RSA') | (
                                df['target'] == 'UberINSupport'))
                   & (df['created_at'] > str(covid_date))]
                , complaints_sentiment_df)
    elif 'pre' in filter_str:
        return (df[((df['target'] == 'Uber_Support') | (df['target'] == 'UberVirgDetroit') | (
                    df['target'] == 'Uber_India') | (df['target'] == 'UberEats') | (df['target'] == 'Uber_Kolkata') | (
                                df['target'] == 'Uber') | (df['target'] == 'UberUKsupport') | (
                                df['target'] == 'Uber_MEX') | (df['target'] == 'Uber_RSA') | (
                                df['target'] == 'UberINSupport'))
                   & (df['created_at'] <= str(covid_date))]
                , complaints_sentiment_df)
    else:
        return (df[((df['target'] == 'Uber_Support') | (df['target'] == 'UberVirgDetroit') | (
                    df['target'] == 'Uber_India') | (df['target'] == 'UberEats') | (df['target'] == 'Uber_Kolkata') | (
                                df['target'] == 'Uber') | (df['target'] == 'UberUKsupport') | (
                                df['target'] == 'Uber_MEX') | (df['target'] == 'Uber_RSA') | (
                                df['target'] == 'UberINSupport'))]
                , complaints_sentiment_df)


@st.cache(persist=True, allow_output_mutation=True)
def calculate_sentiment(df):
    return get_sentiment(df)


date = np.datetime64('2020-04-01T01:00:00.000000+0100')
uber_df = load_data()
uber_df_f, filter_type = filter_data(uber_df.copy(), timeline, direction, date)
network_results = load_network_results(filter_type)
text_df, complaints_sentiment = load_sentiment_results(filter_type, date)
cliques, source_cliques, target_cliques, communities_df = load_data_Clique()
#######################################################################################################

# COMBINED ANALYSIS
#######################################################################################################
st.header('Uber Support Priority Queue')
st.write('Here we combine sentiment and network models to prioritize posts based off priority.')
created_at = st.selectbox(
    "Created At",
    (uber_df['created_at'])
)
score_type = st.selectbox(
    "Select score type (mean favors sentiment & sum favors network)",
    ("Mean", "Sum")
)

sentiment_df = calculate_sentiment(text_df)

combined_df = sentiment_df.merge(network_results, on='user')
scaler = MaxAbsScaler()
if score_type == 'Mean':
    scores = [combined_df['mean_score'], -1*combined_df['sentiment_score']]
    scores_std = scaler.fit_transform(scores)
    combined_df['combined_score'] = (scores_std[0] + scores[1])/2
    combined_df['mean_score_std'] = scores_std[0]
    combined_df['sentiment_score_std'] = scores_std[1]
    combined_df.sort_values('combined_score', ascending=False, inplace=True)
    st.write(combined_df[['user', 'target', 'combined_score', 'mean_score_std', 'sentiment_score_std', 'mean_score', 'sentiment_score', 'text', 'created_at']])
if score_type == 'Sum':
    scores = [combined_df['sum_score'], -1 * combined_df['sentiment_score']]
    scores_std = scaler.fit_transform(scores)
    combined_df['combined_score'] = scores_std[0] + scores[1]
    combined_df['sum_score_std'] = scores_std[0]
    combined_df['sentiment_score_std'] = scores_std[1]
    combined_df.sort_values('combined_score', ascending=False, inplace=True)
    st.write(combined_df[['user', 'target', 'combined_score', 'sum_score_std', 'sentiment_score_std', 'sum_score', 'sentiment_score', 'text', 'created_at']])

#######################################################################################################

# CLIQUES
#######################################################################################################
expander_cliques = st.beta_expander('Cliques Detection')
communities = list(communities_df.groupby('group')['id'].agg(lambda x: ', '.join(x)))
for i in range(len(communities)):
    str_comm = communities[i]
    if len(str_comm.split(',')) > 1:
        expander_cliques.text('Group {}'.format(i+1))
        expander_cliques.text(str_comm)
expander_cliques.subheader("Cliques")
expander_cliques.write(cliques[['source', 'target', 'text', 'clique', 'clique_size']])
expander_cliques.subheader("Source of Cliques")
expander_cliques.write(source_cliques[['Source', 'Counts']])
expander_cliques.subheader("Target of Cliques")
expander_cliques.write(target_cliques[['Target', 'Counts']])
#######################################################################################################

# NETWORK ANALYSIS
#######################################################################################################
expander_network = st.beta_expander('Network Analysis')
expander_network.write(network_results.head(50))
if 'post' in filter_type:
    expander_network.image('./images/post_graph.png')
elif 'pre' in filter_type:
    expander_network.image('./images/pre_graph.png')
else:
    expander_network.image('./images/all_graph.png')
#######################################################################################################

# SENTIMENT ANALYSIS
#######################################################################################################
expander_sentiment = st.beta_expander('Sentiment Analysis')
expander_sentiment.subheader('Sentiment Of Interactions')
expander_sentiment.write(sentiment_df)
expander_sentiment.subheader('Sentiment Based On Complaints Categories')
expander_sentiment.write(complaints_sentiment)

expander_sentiment.subheader('Negative Sentiment Based On Conversation Length')
expander_sentiment.image('./images/senti_by_response_time.png')
#######################################################################################################

# TOPIC MODELING
#######################################################################################################
expander_topics = st.beta_expander('Topic Modeling Analysis')
wc_image = word_cloud(uber_df_f)
expander_topics.image(wc_image, width=1200)

# try:
#  lda(uber_df_f, n_topics, filter_type)
# except IOError as e:
#    if e.errno == errno.EPIPE:
#        print('Waiting on LDA')

HtmlFile = None
if os_type == 'Windows':
    HtmlFile = open(cwd + r"\html\{}_lda_n{}.html".format(filter_type, n_topics), encoding='utf-8')
else:
    HtmlFile = open(cwd + "/html/{}_lda_n{}.html".format(filter_type, n_topics), encoding='utf-8')
source_code = HtmlFile.read()
lda_html = components.html(source_code, width=1200, height=800)
expander_topics.lda_html
#######################################################################################################


