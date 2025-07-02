import streamlit as st
from backend.utils import read_csv_head, create_sentiment_barplot
import matplotlib.pyplot as plt

st.title("CPF SCAN Dashboard")

file_path = st.text_input("Enter the full path to your CSV file:")

if file_path:
    st.write(f"You entered: {file_path}")
    df_head = read_csv_head(file_path)
    st.write(df_head)

    sentiment_plot = create_sentiment_barplot(file_path)
    if isinstance(sentiment_plot, plt.Figure):
        st.pyplot(sentiment_plot)
    else:
        st.write(sentiment_plot)
