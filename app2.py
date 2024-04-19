import streamlit as st
import requests
from pathlib import Path
import pandas as pd
from PIL import Image
import os
from llama_index.core import (
    SimpleDirectoryReader,
    DocumentSummaryIndex,
    StorageContext,
    get_response_synthesizer
)
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.llms.openai import OpenAI
from llama_index.core.indices.document_summary import DocumentSummaryIndexEmbeddingRetriever

# Set up environment
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Helper Functions
def download_wikipedia_pages(titles):
    base_url = "https://en.wikipedia.org/w/api.php"
    for title in titles:
        response = requests.get(
            base_url,
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            }
        ).json()
        page = next(iter(response["query"]["pages"].values()))
        yield title, page["extract"]

def build_document_index(docs):
    chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo")
    doc_summary_index = DocumentSummaryIndex.from_documents(
        docs,
        llm=chatgpt,
        transformations=[],
        response_synthesizer=get_response_synthesizer(response_mode="tree_summarize"),
        show_progress=True
    )
    return doc_summary_index

# Load CSS for styling
def load_css(file_name):
    with open(file_name) as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

# Streamlit app
def main():
    st.set_page_config(
        page_title="Wiki Summary Index",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_css('style.css')
    image = Image.open('logo.png')
    st.sidebar.image(image, width=100)

    st.title("Wiki Summary Index 📚")

    # Downloading and loading data
    if 'document_index' not in st.session_state:
        titles = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]
        docs = []
        for title, text in download_wikipedia_pages(titles):
            docs.append((title, text))
        st.session_state['document_index'] = build_document_index(docs)

    doc_index = st.session_state['document_index']

    # Query interface
    query = st.text_input("Enter your query about city information:")
    if query:
        with st.spinner('Searching for relevant information...'):
            response = doc_index.query(query)
            st.write(response)

if __name__ == "__main__":
    main()
