from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores import FAISS
import PyPDF2

# 1. Extract PDF Text
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Load and prepare both PDF texts
pdf_text_general = extract_text_from_pdf(r"C:\Desktop\Safety_Chatbot\general_safety_rules.pdf")
pdf_text_emergency = extract_text_from_pdf(r"C:\Desktop\Safety_Chatbot\FA-manual.pdf")

# Load the embedding model
embedding_model = HuggingFaceInferenceAPIEmbeddings(
    api_key="hf_your_actual_api_key",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS indexes for both PDFs
faiss_emergency = FAISS.from_texts(pdf_text_emergency, embedding_model)
faiss_general = FAISS.from_texts(pdf_text_general, embedding_model)
