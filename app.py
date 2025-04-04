import os
import json
import re
import pandas as pd
import numpy as np
import nltk
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Medical RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache loading data and preprocessing to avoid redundant computation
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')

@st.cache_data
def extract_text_from_json(data):
    """
    Heuristic to extract the main text content from the JSON data.
    Assumes the most relevant text is the longest string value.
    """
    longest_text = ""
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, str):
                if len(value) > len(longest_text):
                    longest_text = value
            elif isinstance(value, list): # Handle lists of strings
                 list_text = "\n".join(filter(lambda x: isinstance(x, str), value))
                 if len(list_text) > len(longest_text):
                     longest_text = list_text
    elif isinstance(data, str): # If the JSON root is just a string
        longest_text = data
    elif isinstance(data, list): # If the JSON root is a list
        list_text = "\n".join(filter(lambda x: isinstance(x, str), data))
        if len(list_text) > len(longest_text):
            longest_text = list_text

    return longest_text.strip()

@st.cache_data
def load_data_from_structure(data_dir):
   
    all_records = []
    if not os.path.isdir(data_dir):
        return []

    for root, dirs, files in os.walk(data_dir):
        # Check if the current directory seems like a PDD directory (contains .json files)
        if any(f.endswith('.json') for f in files):
            # Try to extract Disease Category and PDD from the path
            try:
                path_parts = os.path.normpath(root).split(os.sep)
                # Expecting structure like [... , data_dir, disease_cat, pdd_cat]
                # Find the index of the base data_dir
                base_dir_index = -1
                norm_data_dir = os.path.normpath(data_dir)
                for i, part in enumerate(path_parts):
                    # Check if the path up to this part matches the base data directory
                    current_path_check = os.path.join(*path_parts[:i+1])
                    if os.path.samefile(current_path_check, norm_data_dir):
                         base_dir_index = i
                         break

                if base_dir_index != -1 and len(path_parts) > base_dir_index + 2:
                    disease_category = path_parts[base_dir_index + 1]
                    pdd_category = path_parts[base_dir_index + 2]
                else:
                    # Fallback if structure is unexpected
                    pdd_category = path_parts[-1] if len(path_parts) > 0 else "Unknown PDD"
                    disease_category = path_parts[-2] if len(path_parts) > 1 else "Unknown Disease"

            except Exception:
                disease_category = "Unknown Disease"
                pdd_category = "Unknown PDD"

            for filename in files:
                if filename.endswith('.json'):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Extract text content using the heuristic
                        text_content = extract_text_from_json(data)

                        if text_content: # Only add if text content is found
                            all_records.append({
                                'id': file_path, # Use file path as a unique ID
                                'disease_category': disease_category,
                                'pdd': pdd_category, # Keep 'pdd' key for consistency downstream
                                'text': text_content
                            })

                    except json.JSONDecodeError:
                        pass
                    except Exception:
                        pass

    return all_records

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple whitespace with single space
    return text

@st.cache_data
def preprocess_documents(documents):
    """
    Applies cleaning and tokenization to the documents.
    Filters out documents with empty cleaned text.
    """
    preprocessed_docs = []
    for doc in documents:
        cleaned_text = clean_text(doc['text'])
        if cleaned_text: # Only keep documents with non-empty text after cleaning
            # Tokenize for BM25 - simple whitespace split is often sufficient,
            # but nltk.word_tokenize is more robust.
            tokens = nltk.word_tokenize(cleaned_text)
            preprocessed_docs.append({
                'id': doc['id'],
                'pdd': doc['pdd'],
                'disease_category': doc.get('disease_category', 'Unknown'),
                'original_text': doc['text'], # Keep original for context generation
                'cleaned_text': cleaned_text,
                'tokens': tokens
            })
    return preprocessed_docs

@st.cache_resource
def build_bm25_index(preprocessed_data):
    """Build BM25 index from preprocessed documents"""
    if not preprocessed_data:
        return None
    
    tokenized_corpus = [doc['tokens'] for doc in preprocessed_data]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

@st.cache_resource
def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None, device

def retrieve_documents(query, bm25_index, preprocessed_docs, top_k=3):
    """
    Retrieves the top_k most relevant documents for a given query using BM25.
    """
    if not bm25_index or not preprocessed_docs:
        return []

    # 1. Preprocess the query (same steps as documents)
    cleaned_query = clean_text(query)
    tokenized_query = nltk.word_tokenize(cleaned_query)

    # 2. Get BM25 scores for the query against all documents
    doc_scores = bm25_index.get_scores(tokenized_query)

    # 3. Get the indices of the top-k documents
    # Ensure we don't request more documents than available
    k = min(top_k, len(preprocessed_docs))
    top_n_indices = np.argsort(doc_scores)[::-1][:k] # Get indices sorted by score descending

    # 4. Retrieve the corresponding documents with scores
    retrieved_docs = []
    for i in top_n_indices:
        if doc_scores[i] > 0:
            doc = preprocessed_docs[i].copy()
            doc['score'] = float(doc_scores[i])
            retrieved_docs.append(doc)

    return retrieved_docs

def generate_answer(query, retrieved_docs, model, tokenizer, device, 
                    max_length=800, min_length=50, num_beams=6, 
                    temperature=0.7, early_stopping=True):
    """
    Generates an answer using the LLM based on the query and retrieved context.
    Uses notebook-like parameters that are proven to work well.
    """
    if not model or not tokenizer:
        return "Error: LLM model or tokenizer not available."
    if not retrieved_docs:
        return "No relevant documents were found to answer the question."

    # Format the context string exactly like in the notebook
    context_str = "\n\n---\n\n".join([doc['original_text'] for doc in retrieved_docs])

    # Use the exact prompt template from the notebook
    prompt_template = """
Based *only* on the following context regarding diagnostic procedures for specific diseases, please answer the question in detail. Provide a comprehensive response covering all relevant information. Do not use any prior knowledge. If the context does not contain the answer, state that the information is not available in the provided documents.

Context:
---
{context_str}
---

Question: {query}

Detailed Answer:
"""
    
    prompt = prompt_template.format(context_str=context_str, query=query)

    # Increase max_length in tokenization to match notebook
    inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True).to(device) 

    # Generate the answer using notebook-like parameters
    try:
        with torch.no_grad(): 
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,           # More beams like in notebook (notebook uses 20)
                temperature=temperature,
                early_stopping=early_stopping,
                no_repeat_ngram_size=3,        # Prevent repetition
                length_penalty=2.0,            # Strongly encourage longer outputs
                repetition_penalty=1.5         # Penalize repetition
            )

        # Decode the output
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Edge case handling
        if len(answer.strip()) < 10 or "I want to" in answer:
            return "The model generated an incomplete response. Try adjusting parameters: increase max_length, decrease temperature, or increase num_beams."
            
        return answer

    except Exception as e:
        return f"Error during generation: {e}"

def main():
    # Add header
    st.title("Medical Diagnostic RAG System")
    st.markdown("Ask questions about medical diagnostic procedures based on the available documents.")
    
    # Initialize app state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.data_loaded = False
        st.session_state.model_name = 'google/flan-t5-base'
    
    st.sidebar.image("image.png", use_container_width=True)

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Data directory input
    data_dir = "samples"
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select LLM Model",
        options=[
            'google/flan-t5-small',
            'google/flan-t5-base',
            'google/flan-t5-large'
        ],
        index=1  # Default to flan-t5-base
    )
    
    # Load data automatically at start
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        with st.spinner("Downloading NLTK data..."):
            download_nltk_data()
            
        with st.spinner(f"Loading data from {data_dir}..."):
            raw_data = load_data_from_structure(data_dir)
            if raw_data:
                st.session_state.raw_data = raw_data
                with st.spinner("Preprocessing documents..."):
                    st.session_state.preprocessed_data = preprocess_documents(raw_data)
                with st.spinner("Building BM25 index..."):
                    st.session_state.bm25 = build_bm25_index(st.session_state.preprocessed_data)
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded {len(raw_data)} documents")
            else:
                st.error(f"❌ No data found in {data_dir}")
    
    
    # Load model button
    if st.sidebar.button("Load Model"):
        with st.spinner(f"Loading {model_name}..."):
            model, tokenizer, device = load_model_and_tokenizer(model_name)
            if model and tokenizer:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.device = device
                st.session_state.model_name = model_name
                st.session_state.model_loaded = True
                st.success(f"✅ Model loaded: {model_name}")
            else:
                st.error("❌ Failed to load model")
    
    # Display data stats if loaded
    if st.session_state.get('data_loaded', False):
        st.sidebar.subheader("Data Statistics")
        df_raw = pd.DataFrame(st.session_state.raw_data)
        st.sidebar.write(f"Total documents: {len(df_raw)}")
        st.sidebar.write(f"Unique PDDs: {df_raw['pdd'].nunique()}")
        st.sidebar.write(f"Unique diseases: {df_raw['disease_category'].nunique()}")
    
    # LLM parameters in sidebar
    st.sidebar.header("LLM Parameters")
    
    top_k = st.sidebar.slider("Number of documents to retrieve (top-k)", 
                             min_value=1, max_value=10, value=3)
    
    max_length = st.sidebar.slider("Max response length", 
                                  min_value=500, max_value=10000, value=3000)
    
    min_length = 10
                                 
    
    num_beams = st.sidebar.slider("Beam search width", 
                                 min_value=1, max_value=20, value=6)
    
    temperature = st.sidebar.slider("Temperature", 
                                   min_value=0.1, max_value=2.0, value=0.7, step=0.1)
    
    early_stopping = True
    
    # Main content area
    if not st.session_state.get('data_loaded', False) or not st.session_state.get('model_loaded', False):
        st.info("Please load model from the sidebar before asking questions.")
    else:
        # User input for question
        user_query = st.text_area("Enter your medical question:", height=100)
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.button("Submit")
        
        if submit_button and user_query:
            with st.spinner("Processing your question..."):
                # Retrieve documents
                retrieved_docs = retrieve_documents(
                    user_query, 
                    st.session_state.bm25, 
                    st.session_state.preprocessed_data, 
                    top_k=top_k
                )
                
                # Generate answer if documents were retrieved
                if retrieved_docs:
                    answer = generate_answer(
                        user_query,
                        retrieved_docs,
                        st.session_state.model,
                        st.session_state.tokenizer,
                        st.session_state.device,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=num_beams,
                        temperature=temperature,
                        early_stopping=early_stopping
                    )
                else:
                    answer = "No relevant documents found to answer your question."
                
                # Display results
                st.subheader("Generated Answer:")
                st.write(answer)
                
                # Display retrieved documents
                st.subheader(f"Top {len(retrieved_docs)} Retrieved Documents:")
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"Document {i+1}: {doc['pdd']} (Score: {doc['score']:.4f})"):
                        st.write(f"**Disease Category:** {doc['disease_category']}")
                        st.write(f"**Document ID:** {os.path.basename(doc['id'])}")
                        st.text_area("Content:", doc['original_text'][:1000] + 
                                     ("..." if len(doc['original_text']) > 1000 else ""), 
                                     height=200, key=f"doc_{i}")

if __name__ == "__main__":
    main()