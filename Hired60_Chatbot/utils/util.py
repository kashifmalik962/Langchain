from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def get_loader(file_path: str, file_type: str):
    """Return appropriate document loader based on file type"""
    if file_type == ".pdf":
        return PyPDFLoader(file_path)
    elif file_type == ".docx":
        return Docx2txtLoader(file_path)
    elif file_type == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")



