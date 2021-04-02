import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import errno
import platform
from Helpers import word_cloud, lda

st.title('Twitter Complaints Analysis')

# SIDEBAR
#######################################################################################################
st.sidebar.image('./images/mcgill_logo.png')

st.sidebar.markdown('**By:** Anukriti, Siddharth, Richard, Bogdan')
st.sidebar.markdown('**McGill MMA** - Social Media Analytics')
st.sidebar.markdown('**GitHub:** https://github.com/bogdan-tanasie/SMF')

st.sidebar.subheader('Options')
timeline = st.sidebar.selectbox(
    "Select timeframe for analysis",
    ("All", "Pre-COVID", "Post-COVID")
)
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


@st.cache(persist=True, allow_output_mutation=True)
def filter_data(df, t, d, covid_date):
    filter_str = 'all'
    if t == 'All' and d == 'To Uber':
        filter_str = 'to'
        return df[df['target'] == 'Uber_Support'], filter_str
    elif t == 'All' and d == 'From Uber':
        filter_str = 'from'
        return df[df['source'] == 'Uber_Support'], filter_str
    elif t == 'Pre-COVID' and d == 'All':
        filter_str = 'pre'
        return df[df['created_at'] <= covid_date], filter_str
    elif t == 'Post-COVID' and d == 'All':
        filter_str = 'post'
        return df[df['created_at'] > covid_date], filter_str
    elif t == 'Pre-COVID' and d == 'To Uber':
        filter_str = 'pre_to'
        return df[(df['target'] == 'Uber_Support') & (df['created_at'] <= covid_date)], filter_str
    elif t == 'Pre-COVID' and d == 'To Uber':
        filter_str = 'pre_from'
        return df[(df['source'] == 'Uber_Support') & (df['created_at'] <= covid_date)], filter_str
    elif t == 'Post-COVID' and d == 'To Uber':
        filter_str = 'post_to'
        return df[(df['target'] == 'Uber_Support') & (df['created_at'] > covid_date)], filter_str
    elif t == 'Post-COVID' and d == 'From Uber':
        filter_str = 'post_from'
        return df[(df['source'] == 'Uber_Support') & (df['created_at'] > covid_date)], filter_str
    else:
        return df, filter_str


date = np.datetime64('2020-04-01T01:00:00.000000+0100')
uber_df = load_data()
uber_df_f, lda_str = filter_data(uber_df.copy(), timeline, direction, date)
#######################################################################################################


# TOPIC MODELING
#######################################################################################################
st.subheader('Topic Modeling Analysis')
wc_image = word_cloud(uber_df_f)
st.image(wc_image, width=1200)

try:
    lda(uber_df_f, n_topics, lda_str)
except IOError as e:
    if e.errno == errno.EPIPE:
        print('Waiting on LDA')

HtmlFile = None
if os_type == 'Windows':
    HtmlFile = open(cwd + r"\html\{}_lda_n{}.html".format(lda_str, n_topics), encoding='utf-8')
else:
    HtmlFile = open(cwd + "/html/{}_lda_n{}.html".format(lda_str, n_topics), encoding='utf-8')
source_code = HtmlFile.read()
components.html(source_code, width=1200, height=800)
#######################################################################################################

