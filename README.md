
```markdown
# NLP Streamlit Web App

## Overview
This Natural Language Processing (NLP) based web app leverages **Hugging Face transformers** and **spaCy** to offer a variety of NLP tasks, including text summarization, sentiment analysis, named entity recognition (NER), text completion, and question answering. It serves as a versatile tool designed to simplify common text-based tasks using state-of-the-art NLP models.

### Technologies Used:
- **Streamlit**: For building the web interface.
- **Hugging Face Transformers**: For text summarization, sentiment analysis, text completion, and question answering.
- **spaCy**: For Named Entity Recognition (NER) using the spaCy transformers pipeline.
- **stqdm**: For displaying progress bars within Streamlit.

## Features

### 1. Text Summarization
- Generate concise summaries of long input text using Hugging Face transformers.
- Customizable by specifying the desired length of the summary.
  
### 2. Sentiment Analysis
- Analyzes the sentiment of the input text (positive, negative, or neutral) using advanced NLP models from Hugging Face.
  
### 3. Named Entity Recognition (NER)
- Identify and extract named entities such as people, organizations, and locations from the input text.
- Powered by **spaCy** transformers for fast and accurate entity recognition.
  
### 4. Text Completion
- Generate completions for partially entered text using Hugging Face’s text-generation pipeline.
  
### 5. Question Answering
- Provides context-based answers to user questions using Hugging Face’s question-answering pipeline.

## Installation

### Prerequisites:
- Python 3.7 or higher

### Setup Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/mvvr/nlp-streamlit-app.git
   cd nlp-streamlit-app
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/macOS
   env\Scripts\activate      # For Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Requirements:
Ensure the following packages are listed in `requirements.txt`:

```txt
streamlit==1.26.0
stqdm==0.0.4
transformers==4.34.0
torch==2.0.1
spacy==3.6.0
spacy-streamlit==1.0.3
pandas==2.1.1
```

## Usage

1. **Text Summarization**: Enter a block of text and specify the desired summary length. The app will generate a concise summary using the Hugging Face summarization model.

2. **Sentiment Analysis**: Paste any text, and the app will analyze its sentiment (positive, negative, or neutral) using Hugging Face's sentiment-analysis pipeline.

3. **Named Entity Recognition (NER)**: Enter text, and the app will highlight named entities like people, locations, and organizations using the spaCy model.

4. **Text Completion**: Enter incomplete text, and the app will generate possible completions.

5. **Question Answering**: Provide a text passage (context) and ask a question related to the passage. The app will extract the answer based on the context provided.

## Screenshots

- **Text Summarization**  
  ![Text Summarization](screenshots/summarization.png)

- **Sentiment Analysis**  
  ![Sentiment Analysis](screenshots/sentiment_analysis.png)

- **Named Entity Recognition (NER)**  
  ![NER](screenshots/ner.png)

## Future Enhancements

- Integrating additional NLP tasks, such as text translation and paraphrasing.
- Adding support for multi-language text processing.
- Expanding text summarization to work with different summarization techniques.

## Contributions

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/nlp-streamlit-app/issues).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Author**: [Your Name](https://github.com/mvvr)  
If you find this project helpful, consider giving it a ⭐ on [GitHub](https://github.com/mvvr/nlp-streamlit-app)!
