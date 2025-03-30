import fitz 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import os

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        # Save the uploaded file to a temporary location
        temp_pdf_path = "temp.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_file.getvalue())
        
        try:
            # Open the PDF file with PyMuPDF
            doc = fitz.open(temp_pdf_path)
            text = ""
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text:  # Check if page text is not empty
                    text += page_text
            
            # Close the document
            doc.close()
            
            # If we didn't get any text, provide a placeholder message
            if not text.strip():
                text = "No text content could be extracted from the PDF."
                
            return text
        except Exception as e:
            # Handle any exceptions that may occur during PDF processing
            print(f"Error extracting text from PDF: {e}")
            return f"Error processing PDF: {str(e)}"
        finally:
            # Always remove the temporary file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
    
    def split_text_into_chunks(self, text: str) -> List[Document]:
        """Split text into smaller chunks for processing"""
        if not text or not text.strip():
            # If the text is empty, return a single document with a message
            return [Document(page_content="The PDF appears to be empty or contains no extractable text.")]
        
        # Create a document with the extracted text
        documents = [Document(page_content=text)]
        
        # Split the document into chunks
        try:
            chunks = self.text_splitter.split_documents(documents)
            # Ensure we have at least one chunk
            if not chunks:
                return [Document(page_content="The PDF content couldn't be properly split into chunks.")]
            return chunks
        except Exception as e:
            # Handle any exceptions that may occur during splitting
            print(f"Error splitting text into chunks: {e}")
            return [Document(page_content=f"Error processing document chunks: {str(e)}")]