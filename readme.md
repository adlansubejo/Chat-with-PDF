# Chat PDF

This repository contains the source code for the Chat PDF project.

## Description

The Chat PDF project is a tool that allows users to convert chat conversations into PDF documents. It provides a simple and convenient way to save and share chat logs in a portable format.

## Features

- Convert chat conversations into PDF documents
- Customize the appearance and formatting of the PDF output
- Support for various chat platforms and formats
- Easy-to-use command-line interface

## Installation

To use this repository on your local machine, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/chat-pdf.git

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt

3. Add .env file with the following content:

    the example of .env file is provided in the .env.example file

    here is the example of .env file:

    ```bash
    HUGGINGFACEHUB_API_TOKEN=hf_xxxxx
    ```

    here is tutorial on how to get API key and secret: [link](https://huggingface.co/docs/hub/security-tokens)

4. Run the application:

    ```bash
    streamlit run streamlit_app.py

## Using Step

1. Open the application in your browser
2. Upload the pdf file
3. wait until chat box appear
4. fill the chat box with your question
5. click process button

