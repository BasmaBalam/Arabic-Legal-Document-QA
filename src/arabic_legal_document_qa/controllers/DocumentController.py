import os 
import re
import json
import subprocess
from typing import Any, Dict, List
from .BaseController import BaseController
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentController(BaseController):

    """Controller responsible for loading and processing legal documents.
    This controller handles locating the processed Egyptian Civil Code document,
    loading its JSON content, splitting articles into logical legal paragraphs,
    and creating LangChain ``Document`` chunks suitable for downstream processing
    such as indexing or retrieval. 
    
    The controller supports both Arabic and English article text and uses 
    language-specific regular expressions to identify legal paragraph markers. 
    
    Attributes:
    
        _DocumentController__ARABIC_PARAGRAPH_MARKER: Regular expression used to identify
        Arabic legal paragraph markers.
        
        _DocumentController__ENGLISH_PARAGRAPH_MARKER: Regular expression used to identify 
        English legal paragraph markers. 
    
    """

    def __init__(self):
        super().__init__()
        self.__ARABIC_PARAGRAPH_MARKER = re.compile(
            r'\(\s*[٠-٩0-9]+\s*'
            r'|[٠-٩0-9]+\s*\)'
        )

        self.__ENGLISH_PARAGRAPH_MARKER = re.compile(
            r'(?m)^\s*\d+\.\s+'
        )

    def get_document_path(self)-> str:

        """Return the path to the processed Egyptian Civil Code JSON file.
        If the expected document does not exist, this method runs ``dvc repro`` 
        to generate the required data artifact. If the document is still unavailable afterward,
        a ``FileNotFoundError`` is raised.
        
        Returns: 
            str: Absolute or configured path to the processed JSON document.
            
        Raises: 
            
            subprocess.CalledProcessError: If ``dvc repro`` fails. 
            FileNotFoundError: If the document cannot be found after attempting to reproduce it.
        """

        json_doc_path = os.path.join(
            self.data_processed_dir,
            "egyptian_civil_code.json"
        )

        if not os.path.exists(json_doc_path):
            subprocess.run(
                ["dvc", "repro"],
                check=True
            )

        if not os.path.exists(json_doc_path):
            raise FileNotFoundError( f"Document was not generated: {json_doc_path}" )

        return json_doc_path

    def get_document_extension(self) -> str:
        
        """Return the file extension of the processed document. 
        Returns: 
            str: The document extension, including the leading dot,
            for example ``".json"``.
        """

        doc_path = self.get_document_path()
        return os.path.splitext(doc_path)[-1]

    def get_document_loader(self) -> JSONLoader:
        
        """Create and return a JSON loader for the processed document.
        Returns:
            JSONLoader: A LangChain JSON loader configured to load the complete JSON document.
         """
        doc_path = self.get_document_path()
        return JSONLoader(file_path=doc_path, jq_schema=".", text_content=False)
 
    def get_document_content(self) -> List[Dict[str, Any]]:
        
        """Load and return the processed document's JSON content.
        The document is expected to contain a list of article dictionaries.
        
        Returns:
            List[Dict[str, Any]]: A list containing the legal articles and their associated metadata.
            
        Raises:
            FileNotFoundError: If the processed document does not exist.
            json.JSONDecodeError: If the document contains invalid JSON. 
        """
        
        doc_path = self.get_document_path()
        with open(doc_path, "r", encoding="utf-8") as file:
            return json.load(file)
 
    def split_into_paragraphs(self, text: str, pattern: re.Pattern) -> List[str]:

        """Split legal text into paragraphs using a regular expression.
        The supplied pattern is used to identify the beginning of legal paragraphs.
        Text appearing before the first paragraph marker is preserved as a separate
        paragraph when it is not empty. If no paragraph markers are found, 
        the entire text is returned as a single paragraph. 
        
        Args:
            text: Legal article text to split.
            pattern: Regular expression used to identify paragraph markers.
            
        Returns:
            List[str]: A list of cleaned legal paragraphs.
            
        """
        matches = list(pattern.finditer(text))

        if not matches:
            return [text.strip()]

        paragraphs = []

        lead = text[:matches[0].start()].strip()

        if lead:
            paragraphs.append(lead)

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            para = text[start:end].strip()
            if para:
                paragraphs.append(para)
 
        return paragraphs

    def create_chunks(self, text: str, metadata: Dict[str, Any], chunk_size: int, splitter: RecursiveCharacterTextSplitter, pattern: re.Pattern) -> List[Document]:
        """Create LangChain documents from a legal text.
        The method uses a two-stage chunking strategy: 
            
            1. If the complete text fits within ``chunk_size``, it is returned as 
            a single document.
            
            2. Otherwise, the text is first split according to legal paragraph markers. 
            Paragraphs exceeding ``chunk_size`` are further split using the provided 
            ``RecursiveCharacterTextSplitter``.
            
            Each resulting document receives the supplied metadata along with chunk position information.
            
            Args:
                text: Legal text to split into chunks.
                metadata: Metadata to attach to every generated document.
                chunk_size: Maximum preferred size of a chunk.
                splitter: LangChain text splitter used for oversized paragraphs.
                pattern: Regular expression used to identify legal paragraphs.
                
            Returns: 
                List[Document]: Generated LangChain document chunks. 
                Returns an empty list when the input text is empty.
            """
        text = (text or "").strip()

        if not text:
            return []

        # Case 1: article fits in a single chunk
        if len(text) <= chunk_size:
            return [ Document(
                    page_content=text,
                    metadata={**metadata,"chunk_part": 1,"chunk_count": 1},
                )
            ]

        # Case 2: split using the article's legal paragraphs
        paragraphs = self.split_into_paragraphs(text, pattern)

        chunks = []
        chunk_count = len(paragraphs)
        
        for i, paragraph in enumerate(paragraphs, start=1):

            # If paragraph is still too large, use character splitter
            if len(paragraph) > chunk_size:
                parts = splitter.split_text(paragraph)
            else:
                parts = [paragraph]

            for j, part in enumerate(parts, start=1):
                chunks.append(
                    Document(
                        page_content=part,
                        metadata={**metadata,"chunk_part": i, "chunk_count": chunk_count},
                    )
                )

        return chunks

    def process_document_content(self, doc_content: List[Dict[str, Any]], chunk_size: int = 1000, overlap_size: int= 100) -> List[Document]:
        """Process legal articles and convert them into document chunks.
        Each article may contain Arabic and English versions of the legal text.
        Both versions are processed independently using their respective paragraph
        markers and are tagged with a language-specific metadata field.
        
        The following article metadata is preserved: 
        
            - ``article_number``
            - ``book`` 
            - ``chapter``
            - ``section``
            - ``topic``
            - ``citation``
            - ``is_repealed``
            - ``source_page`` 
            
        Args:
            doc_content: List of legal articles represented as dictionaries.
            chunk_size: Maximum preferred size of generated text chunks. 
                Defaults to ``1000``.
            overlap_size: Number of overlapping characters between chunks when the recursive 
                text splitter is used. Defaults to ``100``.
            
        Returns: 
            List[Document]: All generated Arabic and English document chunks.
        
        Raises:
            ValueError: If ``chunk_size`` or ``overlap_size`` contains an invalid value accepted by the underlying splitter.
        """
        
        chunks: List[Document] = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            separators=["\n", ". ", "، ", " "],
        )

        for article in doc_content:

            text_ar = (article.get("text_ar") or "").strip()
            text_en = (article.get("text_en") or "").strip()

            base_metadata = {
                "article_number": article.get("article_number"),
                "book": article.get("book"),
                "chapter": article.get("chapter"),
                "section": article.get("section"),
                "topic": article.get("topic"),
                "citation": article.get("citation"),
                "is_repealed": article.get("is_repealed", False),
                "source_page": article.get("source_page"),
            }

            # Arabic
            ar_metadata = {
                **base_metadata,
                "language": "ar",
            }

            chunks.extend(
                self.create_chunks(
                    text=article.get("text_ar", ""),
                    metadata=ar_metadata,
                    chunk_size=chunk_size,
                    splitter=splitter,
                    pattern= self.__ARABIC_PARAGRAPH_MARKER
                )
            )

            # English
            en_metadata = {
                **base_metadata,
                "language": "en",
            }

            chunks.extend(
                self.create_chunks(
                    text=article.get("text_en", ""),
                    metadata=en_metadata,
                    chunk_size=chunk_size,
                    splitter=splitter,
                    pattern= self.__ENGLISH_PARAGRAPH_MARKER
                )
            )

        return chunks
