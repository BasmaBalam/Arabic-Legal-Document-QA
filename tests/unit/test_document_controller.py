import re

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from arabic_legal_document_qa.controllers.DocumentController import DocumentController

ENGLISH_PARAGRAPH_MARKER = re.compile(r"(?m)^\s*\d+\.\s+")


@pytest.fixture
def controller():
    """Create a controller without loading application settings."""
    return DocumentController.__new__(DocumentController)


def test_split_into_paragraphs_preserves_markers(controller):
    result = controller.split_into_paragraphs(
        "Introduction\n1. First paragraph\n2. Second paragraph",
        ENGLISH_PARAGRAPH_MARKER,
    )

    assert result == [
        "Introduction",
        "1. First paragraph",
        "2. Second paragraph",
    ]


def test_create_chunks_returns_empty_list_for_empty_text(controller):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=0,
    )

    result = controller.create_chunks(
        text="   ",
        metadata={"article_number": 1},
        chunk_size=20,
        splitter=splitter,
        pattern=ENGLISH_PARAGRAPH_MARKER,
    )

    assert result == []


def test_create_chunks_preserves_metadata(controller):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
    )

    result = controller.create_chunks(
        text="Legal article text",
        metadata={"article_number": 147, "language": "en"},
        chunk_size=100,
        splitter=splitter,
        pattern=ENGLISH_PARAGRAPH_MARKER,
    )

    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].page_content == "Legal article text"
    assert result[0].metadata["article_number"] == 147
    assert result[0].metadata["language"] == "en"
    assert result[0].metadata["chunk_part"] == 1
    assert result[0].metadata["chunk_count"] == 1
