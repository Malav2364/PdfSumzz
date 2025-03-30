import streamlit as st
import os
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Initialize PDF processor
pdf_processor = PDFProcessor()

def initialize_chain(pdf_docs):
    """Initialize the conversation chain with the provided PDF documents"""
    try:
        # Extract and process the text from the PDF
        text = pdf_processor.extract_text_from_pdf(pdf_docs)
        chunks = pdf_processor.split_text_into_chunks(text)
        
        # Initialize the Mistral AI model
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral-large-latest"
        )
        
        # Create embeddings and vector store using HuggingFace
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create the vector store from the document chunks
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Use a simpler RetrievalQA chain instead of ConversationalRetrievalChain
        # This avoids the complexity of conversation memory and should work more reliably
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain, None
    except Exception as e:
        # If an error occurs, return None for the chain and the error message
        error_message = f"Error initializing chain: {str(e)}"
        print(error_message)
        return None, error_message

# Streamlit app
def main():
    st.set_page_config(
        page_title="PDF Reader with AI Q&A",
        page_icon="📚"
    )
    
    st.title("📚 PDF Reader with AI Q&A")
    st.write("Upload a PDF and ask questions about its content")
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "error" not in st.session_state:
        st.session_state.error = None
    
    # Add a sidebar with info about Malav2364
    st.sidebar.title("About")
    st.sidebar.write("Created by: Malav2364")
    st.sidebar.write("Date: 2025-03-30")
    st.sidebar.write("This app uses LangChain, PyMuPDF, and Mistral AI to analyze PDFs and answer questions about them.")
    
    # Check if Mistral API key is set
    if not os.getenv("MISTRAL_API_KEY"):
        st.warning("Mistral API key not found. Please add your API key to the .env file.")
        st.code("MISTRAL_API_KEY=your_api_key_here", language="text")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Initialize the conversation chain if it hasn't been created yet
        if st.session_state.conversation is None:
            with st.spinner("Processing PDF..."):
                chain, error = initialize_chain(uploaded_file)
                st.session_state.conversation = chain
                st.session_state.error = error
                
            if st.session_state.conversation:
                st.success("PDF processed successfully! You can now ask questions.")
            else:
                st.error(f"Failed to process PDF: {st.session_state.error}")
        
        # Create the chat interface if conversation was initialized successfully
        if st.session_state.conversation is not None:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Chat input
            user_query = st.chat_input("Ask a question about the PDF:")
            if user_query:
                # Add user message to chat history
                user_message = {"role": "user", "content": user_query}
                st.session_state.chat_history.append(user_message)
                
                # Display user message
                with st.chat_message("user"):
                    st.write(user_query)
                
                # Display AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # For RetrievalQA, we use "query" instead of "question"
                            response = st.session_state.conversation({"query": user_query})
                            
                            # Extract the answer from the response
                            answer = response.get("result", "I couldn't generate an answer. Please try a different question.")
                            st.write(answer)
                            
                            # Add AI response to chat history
                            ai_message = {"role": "assistant", "content": answer}
                            st.session_state.chat_history.append(ai_message)
                        except Exception as e:
                            error_msg = f"Error generating response: {str(e)}"
                            st.error(error_msg)
                            
                            # Add error message to chat history
                            ai_message = {"role": "assistant", "content": f"Sorry, I encountered an error: {error_msg}"}
                            st.session_state.chat_history.append(ai_message)
    else:
        st.info("Please upload a PDF file to get started.")
        
        # Add sample questions
        st.subheader("Example questions you can ask:")
        st.write("• What is the main topic of this document?")
        st.write("• Can you summarize the content?")
        st.write("• What are the key findings in this paper?")
        st.write("• Explain the methodology used in this document.")
        st.write("• What conclusions are drawn in this document?")

if __name__ == "__main__":
    main()