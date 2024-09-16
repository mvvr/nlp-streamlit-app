import streamlit as st
from time import sleep
from stqdm import stqdm
import pandas as pd
from transformers import pipeline
import json
import spacy
import spacy_streamlit

# Sidebar function for app introduction
def show_sidebar_info(key):
    st.sidebar.write(
        """
        # NLP Web Application
        
        This app offers several NLP functions, such as sentiment analysis, text completion, summarization, and more.
        
        Built using pretrained transformer models, it efficiently processes and manipulates textual input.
        
        ## Features:
        - Advanced Text Summarizer
        - Named Entity Recognition
        - Sentiment Analysis
        - Question Answering with Context
        - Text Completion
        """
    )

# Cache models to avoid reloading on every call
@st.cache_resource
def load_summarization_model():
    return pipeline('summarization', framework='pt')

@st.cache_resource
def load_ner_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering")

@st.cache_resource
def load_text_completion_model():
    return pipeline("text-generation")


# Main app function
def main():
    st.title("NLP Web Application")
    
    # Sidebar menu
    options = [
        "--Select--", "Summarizer", "Named Entity Recognition", 
        "Sentiment Analysis", "Question Answering with Context", "Text Completion"
    ]
    choice = st.sidebar.selectbox("Choose a feature:", options)
    
    # Introduction if no option is selected
    if choice == "--Select--":
        st.write("""
            Welcome to this NLP-based application that leverages transformer pipelines for various text-based tasks.
            NLP (Natural Language Processing) is a subfield of AI that enables machines to understand and manipulate human language.
        """)
        st.image('NLP.webp', caption='NLP Overview')

    # Summarizer feature
    elif choice == "Summarizer":
        st.subheader("Text Summarization")
        raw_text = st.text_area("Enter the text you want to summarize:")
        min_words = st.number_input("Enter minimum summary length:", min_value=10, value=30, step=5)
        max_words = st.number_input("Enter maximum summary length:", min_value=50, value=100, step=10)
        
        if raw_text and max_words >= min_words:
            try:
                summarizer = load_summarization_model()
                summary = summarizer(raw_text, min_length=min_words, max_length=max_words)
                summary_text = json.loads(json.dumps(summary[0]))['summary_text']
                st.write(f"**Summary**: {summary_text.capitalize()}")
            except Exception as e:
                st.error(f"Error during summarization: {str(e)}")

    # Named Entity Recognition (NER) feature
    elif choice == "Named Entity Recognition":
        st.subheader("Named Entity Recognition (NER)")
        raw_text = st.text_area("Enter text to extract named entities:")
        
        if raw_text:
            try:
                nlp = load_ner_model()
                doc = nlp(raw_text)
                stqdm(range(50), desc="Processing NER...")
                spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title="Identified Entities")
            except Exception as e:
                st.error(f"Error during NER processing: {str(e)}")

    # Sentiment Analysis feature
    elif choice == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        raw_text = st.text_area("Enter text to analyze sentiment:")
        
        if raw_text:
            try:
                sentiment_analysis = load_sentiment_model()
                result = sentiment_analysis(raw_text)[0]
                sentiment = result['label']
                stqdm(range(50), desc="Analyzing sentiment...")
                
                if sentiment == "POSITIVE":
                    st.write("Sentiment: **Positive** 😄")
                elif sentiment == "NEGATIVE":
                    st.write("Sentiment: **Negative** 😟")
                else:
                    st.write("Sentiment: **Neutral** 😐")
            except Exception as e:
                st.error(f"Error during sentiment analysis: {str(e)}")

    # Question Answering with Context feature
    elif choice == "Question Answering with Context":
        st.subheader("Question Answering")
        context = st.text_area("Enter context:")
        question = st.text_area("Enter your question:")
        
        if context and question:
            try:
                question_answering = load_qa_model()
                answer = question_answering(question=question, context=context)['answer']
                st.write(f"**Answer**: {answer.capitalize()}")
            except Exception as e:
                st.error(f"Error during question answering: {str(e)}")

    # Text Completion feature
    elif choice == "Text Completion":
        st.subheader("Text Completion")
        incomplete_text = st.text_area("Enter incomplete text to generate:")
        
        if incomplete_text:
            try:
                text_generation = load_text_completion_model()
                generated = text_generation(incomplete_text)[0]['generated_text']
                st.write(f"**Completed Text**: {generated.capitalize()}")
            except Exception as e:
                st.error(f"Error during text generation: {str(e)}")

if __name__ == "__main__":
    show_sidebar_info("sidebar")
    main()
