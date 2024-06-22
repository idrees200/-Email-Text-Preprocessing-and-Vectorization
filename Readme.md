# Email Text Preprocessing and Vectorization

This project focuses on preprocessing email text data to prepare it for machine learning tasks such as classification, clustering, or sentiment analysis. The preprocessing steps include cleaning text data, removing emojis, stemming, removing stop words, and converting the cleaned text into numerical representations using TF-IDF and Bag-of-Words (BoW) techniques.

## Overview

The provided Python script demonstrates how to preprocess email text data using various techniques available in natural language processing (NLP). The goal is to transform raw text from emails into a format suitable for machine learning models. The steps involved are:

1. **Loading Data:** The script loads the email dataset (`emails.csv`) into a pandas DataFrame for processing.

2. **Text Cleaning:** It cleans the text by removing HTML tags, special characters, and extra whitespaces using regular expressions and BeautifulSoup.

3. **Emoji Removal:** Emojis are removed from the email body text using a regex pattern targeting Unicode emojis.

4. **Stemming:** Words are stemmed to reduce them to their root form using the Porter Stemmer algorithm from NLTK.

5. **Stop Words Removal:** Common English stop words (like "the", "is", "at", etc.) are removed to focus on meaningful words.

6. **Tokenization:** The cleaned text is tokenized into individual words to prepare for vectorization.

7. **Vectorization:** The script converts the cleaned text into numerical vectors using two techniques:
   - **TF-IDF (Term Frequency-Inverse Document Frequency):** Measures the importance of a word in the document relative to the corpus.
   - **Bag-of-Words (BoW):** Represents text as a frequency distribution of words.

8. **Data Representation:** The processed data (`cleaned_text`, `stemmed_email`, `no_stop_words_email`, `tokenized_email`) and their corresponding vectorized representations (`tfidf_vectors`, `bow_vectors`) are displayed and/or saved depending on the script configuration.

## Dependencies

To run the script, you need the following dependencies:

- Python 3.x
- pandas
- numpy
- re
- BeautifulSoup (bs4)
- nltk
- scikit-learn (sklearn)

You can install the required packages using pip:

```bash
pip install pandas numpy beautifulsoup4 nltk scikit-learn
