import streamlit as st

st.sidebar.image('./images/mcgill_logo.png')

st.title('Twitter Complaints Analysis')
st.sidebar.markdown('**By:** Anukriti, Siddharth, Richard, Bogdan')
st.sidebar.markdown('**McGill MMA** - Social Media Analytics')
st.sidebar.markdown('**GitHub:** https://github.com/bogdan-tanasie/SMF')

@st.cache(persist=True)
def load_data():
    return None