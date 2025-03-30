# PDF Sumzz

**PDF Sumzz** is a powerful application that allows users to upload PDF documents and ask questions about their content. The AI analyzes the document and provides accurate, contextual answers based on the information in the PDF.

## Created by
**Malav Patel**  

## Features

- **PDF Upload**: Upload any PDF document for AI analysis
- **AI-Powered Q&A**: Ask questions about the document and receive intelligent answers
- **Context-Aware Responses**: The AI understands the context within the document
- **Interactive Chat Interface**: User-friendly chat interface for seamless interaction
- **Document Chunking**: Efficiently processes large documents by splitting them into manageable pieces
- **Vector Search**: Uses embeddings and vector search to find the most relevant information

## Technologies Used

- **LangChain**: For orchestrating the AI workflow and document processing
- **PyMuPDF (Fitz)**: For extracting text from PDF documents
- **Mistral AI**: For generating intelligent responses to questions
- **FAISS**: For vector similarity search
- **Streamlit**: For creating the web interface
- **Sentence Transformers**: For generating vector embeddings

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Malav2364/PdfSumzz.git
    cd PdfSumzz
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv

    # Activate the virtual environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the project root directory and add your API key:
    ```text
    MISTRAL_API_KEY=your_mistral_api_key_here
    ```

## Usage

1. Start the application:
    ```bash
    streamlit run app.py
    ```

2. Open the application in your web browser (typically at http://localhost:8501)

3. Upload a PDF document using the file uploader

4. Once the document is processed, ask questions about its content

5. The AI will provide answers based on the document's content

## Example Questions

- What is the main topic of this document?
- Can you summarize the content?
- What are the key findings in this paper?
- Explain the methodology used in this document.
- What conclusions are drawn in this document?

## Project Structure

PdfSumzz/ ├── .env # For API keys (not included in repository) ├── app.py # Main Streamlit application ├── pdf_processor.py # PDF processing logic ├── requirements.txt # Project dependencies └── README.md # Project documentation

