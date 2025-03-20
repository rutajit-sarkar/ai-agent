import streamlit as st
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool  # Corrected Integration
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
import os
import numpy as np

# Initialize the Streamlit UI
st.title("Groq Llama3 RAG Chatbot")

# Load the data
uploaded_file = st.file_uploader("Upload your Excel/CSV file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Load data with pandas
        if uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, encoding='utf-8')

        st.dataframe(df)

        # Generate embeddings from text data
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Combine all text columns into one string per row for embeddings
        text_data = df.astype(str).apply(lambda row: ' '.join(row), axis=1).tolist()

        embeddings = model.encode(text_data, show_progress_bar=False)

        st.success("Embeddings generated successfully!")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Create FAISS index using `.from_texts()`
    embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(text_data, embedding=embedding_function)

    # Wrap retriever as a Tool
    retriever_tool = Tool(
        name="Data Retriever",
        func=vectorstore.as_retriever().get_relevant_documents,
        description="Retrieves relevant data from the uploaded file."
    )

    # Create the LLM chain using Groq API
    template = PromptTemplate.from_template(
        "Question: {question}\nAnswer: {answer}"
    )

    groq_key = ""

    if groq_key:
        try:
            # Correct Groq Llama3 Integration
            llm = ChatGroq(
                model="llama3-8b-8192",
                temperature=0.2,
                max_tokens=2000,
                api_key=groq_key
            )

            # Initialize LLMChain with Runnable LLM
            chain = LLMChain(
                llm=llm,
                prompt=template,
                verbose=True
            )

            # Initialize the correct agent
            agent = initialize_agent(
                tools=[retriever_tool],
                agent_type="conversational-react-description",
                llm=llm,
                verbose=True
            )

            # Chat interface
            st.header("Chat with the RAG Chatbot")
            question = st.text_input("Enter your question about the data...")
            if question:
                response = agent.run(question)
                st.write(response)

        except Exception as e:
            st.error(f"Error initializing LLMChain: {e}")
