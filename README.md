# Medical RAG System - README

## Overview

This Medical RAG (Retrieval-Augmented Generation) System is a Streamlit application that leverages Natural Language Processing techniques to answer questions about medical diagnostic procedures. The system retrieves relevant documents from a medical corpus and uses a language model to generate detailed answers based on the retrieved content.

## Features

- **Document Retrieval**: Uses BM25 algorithm to find the most relevant medical documents for a given query
- **Answer Generation**: Employs Flan-T5 models to generate comprehensive answers based on retrieved documents
- **Interactive UI**: Streamlit-based interface with configuration options and result visualization
- **Multiple Model Options**: Choose between different sizes of Flan-T5 models based on your hardware capabilities

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- NLTK
- Streamlit
- rank_bm25
- pandas
- numpy

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Structure

The application expects medical documents in a specific directory structure:

```
samples/
├── Disease Category 1/
│   ├── PDD Category 1/
│   │   ├── document1.json
│   │   ├── document2.json
│   │   └── ...
│   └── PDD Category 2/
│       ├── document1.json
│       └── ...
├── Disease Category 2/
└── ...
```

Each JSON file should contain medical diagnostic procedure information. The system will automatically extract text content from these files.

## Running the Application

1. Ensure your medical documents are placed in the samples directory (or modify the `data_dir` variable in app.py to point to your data location)

2. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. The application will open in your default web browser (typically at http://localhost:8501)

## Using the Application

1. **Initial Setup**:
   - The application automatically loads documents from the specified directory on startup
   - Click "Load Model" in the sidebar to initialize the language model

2. **Configure Parameters** (optional):
   - Select a model size from the dropdown (smaller models are faster but less accurate)
   - Adjust retrieval and generation parameters using the sliders in the sidebar

3. **Ask Questions**:
   - Type your medical diagnostic question in the text area
   - Click "Submit" to process your question

4. **Review Results**:
   - The system will display a generated answer based on the retrieved documents
   - Expand document sections to view the source information used to generate the answer

## Model Selection

- **flan-t5-small**: Fastest option, suitable for systems with limited resources
- **flan-t5-base**: Balanced option (default)
- **flan-t5-large**: Most accurate but requires more memory and processing power

## Troubleshooting

- If you encounter CUDA memory issues, try using a smaller model
- Ensure your JSON documents are properly formatted
- For large document collections, initial loading may take some time

## License

[Include your license information here]

## Acknowledgments

This application uses the following open-source libraries and models:
- Streamlit for the user interface
- Hugging Face Transformers for language models
- rank_bm25 for document retrieval
- NLTK for text processing