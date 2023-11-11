# Pregnancy-CHATBOT
This project is aimed at helping answering the most frequently asked questions about pregnancies 

## Overview

This is a Streamlit-powered chatbot application designed to answer common questions related to pregnancy. The chatbot is powered by LangChain and uses an OpenAI language model for question answering.

## Prerequisites

Make sure you have the following installed on your system:

- Python (>= 3.9)
- Pip (Python package installer)

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/AmosMaru/Pregnancy-CHATBOT.git
    ```

2. Navigate to the project directory:

    ```bash
    cd pregnancy-chatbot
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    ```

    Activate the virtual environment:

    - On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

    - On Unix or MacOS:

        ```bash
        source venv/bin/activate
        ```

5. Create a file named `.env` in the project directory and add your OpenAI API key:

    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    ```

    Replace `your_openai_api_key_here` with your actual OpenAI API key.

## Running the App

```bash
streamlit run app.py
