import os
import re
import json
from pypdf import PdfReader
from typing import Optional, Tuple, List, Dict, Any
from arabic_legal_document_qa.configs.config import Settings, get_settings

from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Input PDF
pdf_path = (PROJECT_ROOT/ "data"/ "raw"/ "egyptian_civil_code.pdf")
# Output JSON
output_path = (PROJECT_ROOT/ "data"/ "processed"/ "egyptian_civil_code.json")

class LegalTextParser: 
    """ 
    Parser for bilingual legal text extracted from PDFs.

    Handles: 
    - Arabic and English article numbers 
    - Arabic-Indic digit conversion 
    - Repealed article detection 
    - Article finalization 
    - Arabic/English bilingual article pairing 
    - Legal hierarchy tracking 
    - PDF page parsing
    """

    def to_int(self, val_str: str, is_arabic: bool = False) -> int:

        """
        Convert Arabic-Indic or Western digits to an integer.

        This function is mainly used for article numbers extracted from
        the Egyptian Civil Code PDF.

        Arabic-Indic digits:
            ١٢٣ -> 123

        Western digits:
            123 -> 123

        For Arabic RTL extraction, the extracted digit sequence may appear reversed. In that case, the digit sequence is reversed before
        converting it to a standard integer.

        Args:
            value: String containing the number to convert.
            is_arabic: Whether the value was extracted from Arabic text.

        Returns:
            The converted integer.
        """
        
        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        cleaned_digits = [char for char in val_str if char.isdigit() or char in arabic_digits]

        if not cleaned_digits:
            return 0

        has_arabic_digits = any(char in arabic_digits for char in cleaned_digits)

        if is_arabic or has_arabic_digits:
            cleaned_digits = cleaned_digits[::-1]

        digit_str = "".join(cleaned_digits)

        for idx, digit in enumerate(arabic_digits):
            digit_str = digit_str.replace(digit, str(idx))

        return int(digit_str)

    def detect_repealed(self, text: str) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Detect a repeal notice and extract its article range.

        Supported English examples:
            Articles 54-80 have been repealed
            Articles 54–80 repealed
            Articles 54 to 80 have been repealed

        Args:
            text: Text extracted from the PDF.

        Returns:
            A tuple containing:
                - is_repealed: True if a repeal notice is detected.
                - start_article: First article in the repealed range,
                or None if no range is found.
                - end_article: Last article in the repealed range,
                or None if no range is found.
        """
        
        is_repealed = any(
            word.lower() in text.lower()
            for word in ["ألغيت", "ملغاة", "repealed"]
        )

        if not is_repealed:
            return False, None, None

        range_match = re.search(r"Articles?\s*(\d+)\s*(?:[-–]|to)\s*(\d+)", text, re.IGNORECASE)

        if range_match:
            n1 = self.to_int(range_match.group(1), is_arabic=False)
            n2 = self.to_int(range_match.group(2), is_arabic=False)
            start_num, end_num = min(n1, n2), max(n1, n2)
            return True, start_num, end_num

        return True, None, None
    def finalize_article(self, art_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert an active article object into the final JSON structure.

        The function joins Arabic and English lines into complete strings
        and preserves the article's hierarchy and source page.

        Args:
            art_dict: Dictionary representing the article currently being
                built by the parser.

        Returns:
            A dictionary ready to be stored in the final JSON corpus.
        """
        
        text_ar = " ".join(art_dict["ar_lines"]).strip()
        text_en = " ".join(art_dict["en_lines"]).strip()

        return {
            "article_number": art_dict["article_number"],
            "book": art_dict["book"],
            "chapter": art_dict["chapter"],
            "section": art_dict["section"],
            "topic": art_dict["topic"],
            "text_ar": text_ar,
            "text_en": text_en,
            "is_repealed": False,
            "source_page": art_dict["source_page"],
            "citation": f"Egyptian Civil Code, Article {art_dict['article_number']}",
        }

    def is_bilingual_completion(
        self, 
        current_article: Optional[Dict[str, Any]],
        is_arabic_header: bool,
    ) -> bool:
        
        """
        Determine whether a newly detected article header belongs to
        the other language of the currently active article.

        This is needed because the PDF may represent one article as:

            مادة ١
            Arabic text...

            Article 1
            English text...

        In this case, "Article 1" is not a new article. It is the
        English half of the same article.

        Args:
            current_article: The article currently being built.
            is_arabic_header: True when the newly detected header is Arabic.

        Returns:
            True if the header is the missing language half of the current article; otherwise False.
        """
        if current_article is None:
            return False
        has_ar = bool(current_article["ar_lines"])
        has_en = bool(current_article["en_lines"])
        return (has_en and not has_ar) if current_article else (has_ar and not has_en)

    def parse_legal_text(
        self, 
        pages_data: List[Tuple[int, str]],
    ) -> List[Dict[str, Any]]:
        
        """
        Parse extracted PDF pages into structured legal articles.

        The parser:
            - tracks book/chapter/section/topic hierarchy,
            - detects Arabic and English article headers,
            - pairs Arabic and English content belonging to the same article,
            - detects repealed article ranges,
            - preserves the source page for each article.

        Args:
            pages_data: A list of tuples in the form:
                (page_number, page_text)

        Returns:
            A list of structured article dictionaries ready for JSON export.
        """

        articles = []

        current_hierarchy = {
            "book": "Preliminary Provisions",
            "chapter": "General Provisions",
            "section": None,
            "topic": None,
        }

        re_book = re.compile(r"^BOOK\s+[IVXLCDM]+$", re.IGNORECASE)
        re_chapter = re.compile(r"^CHAPTER\s+[IVXLCDM]+$", re.IGNORECASE)
        re_section = re.compile(r"^SECTION\s+[IVXLCDM]+$", re.IGNORECASE)
        re_topic = re.compile(r"^(?:[0-9٠-٩]+\s*[-–]\s*.*?\s+)?\d+\.\s+.+$", re.UNICODE)

        re_ar_article = re.compile(r"^مادة\s*[\(\s]*([٠-٩0-9\s]+)", re.UNICODE)
        re_en_article = re.compile(r"^Article\s*(\d+)", re.IGNORECASE)

        current_article = None

        for page_num, text in pages_data:

            lines = [line.strip() for line in text.split("\n") if line.strip()]
            i = 0
            while i < len(lines):
                line = lines[i]

                is_repealed, start_num, end_num = self.detect_repealed(line)
                
                if is_repealed:
                    if start_num is not None and end_num is not None:
                        for num in range(start_num, end_num + 1):
                            articles.append({
                                "article_number": num,
                                "book": current_hierarchy["book"],
                                "chapter": current_hierarchy["chapter"],
                                "section": current_hierarchy["section"],
                                "topic": current_hierarchy["topic"],
                                "text_ar": "",
                                "text_en": "",
                                "is_repealed": True,
                                "source_page": page_num,
                                "citation": f"Egyptian Civil Code, Article {num}",
                            })
                    i += 1
                    continue

                if re_book.match(line):
                    if current_article:
                        articles.append(self.finalize_article(current_article))
                        current_article = None
                    if i + 2 < len(lines):
                        current_hierarchy["book"] = lines[i + 1]
                        i += 2
                        continue

                elif re_chapter.match(line):
                    if current_article:
                        articles.append(self.finalize_article(current_article))
                        current_article = None
                    if i + 2 < len(lines):
                        current_hierarchy["chapter"] = lines[i + 1]
                        i += 2
                        continue

                elif re_section.match(line):
                    if current_article:
                        articles.append(self.finalize_article(current_article))
                        current_article = None
                    if i + 2 < len(lines):
                        current_hierarchy["section"] = lines[i + 1]
                        i += 2
                        continue

                elif re_topic.match(line):
                    if current_article:
                        articles.append(self.finalize_article(current_article))
                        current_article = None
                    en_topic_match = re.search(r'(\d+\.\s+[A-Za-z].*|[A-Za-z].*)', line)
                    current_hierarchy["topic"] = (
                        en_topic_match.group(1).strip() if en_topic_match else line.strip()
                    )
                    i += 1
                    continue

                ar_match = re_ar_article.match(line)
                en_match = re_en_article.match(line)

                if ar_match or en_match:
                    is_ar_header = bool(ar_match)
                    article_num = (
                        self.to_int(ar_match.group(1), is_arabic=True)
                        if ar_match
                        else self.to_int(en_match.group(1), is_arabic=False)
                    )

                    if current_article and current_article["article_number"] == article_num:
                        # Same number reported again (continuation / repeated marker).
                        i += 1
                        continue

                    if self.is_bilingual_completion(current_article, is_ar_header):
                        # This header completes the article we're already building.
                        # Trust the English (Western-digit) number over the Arabic one.
                        if not is_ar_header and current_article["article_number"] != article_num:
                            current_article["article_number"] = article_num
                        i += 1
                        continue

                    # Genuinely a new article.
                    if current_article:
                        articles.append(self.finalize_article(current_article))

                    current_article = {
                        "article_number": article_num,
                        "book": current_hierarchy["book"],
                        "chapter": current_hierarchy["chapter"],
                        "section": current_hierarchy["section"],
                        "topic": current_hierarchy["topic"],
                        "ar_lines": [],
                        "en_lines": [],
                        "source_page": page_num,
                    }
                    i += 1
                    continue

                if current_article:
                    if re.search(r"[\u0600-\u06FF]", line):
                        current_article["ar_lines"].append(line)
                    else:
                        current_article["en_lines"].append(line)

                i += 1

        if current_article:
            articles.append(self.finalize_article(current_article))

        return articles


if __name__ == "__main__":

    parser = LegalTextParser()
    reader = PdfReader(pdf_path)
    settings = get_settings()
    pages_data = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages_data.append((page_number, text))

    extracted_articles = parser.parse_legal_text(pages_data)

    # Remove false article entries that contain no Arabic text,
    # but keep repealed articles because their text is intentionally empty.

    extracted_articles = [
        a for a in extracted_articles
        if a["text_ar"] or a["is_repealed"]
    ]

    print(f"Successfully extracted {len(extracted_articles)} articles.")

    if extracted_articles:
        print(json.dumps(extracted_articles[0], ensure_ascii=False, indent=2))

    # QA pass: flag anything still worth a manual look. Computed after the
    # fact rather than stored on each record, so it doesn't change the
    # output schema your downstream embedding/RAGAS steps expect.
    flagged = [
        a for a in extracted_articles
        if not a["text_ar"] or not a["text_en"]
        or a["article_number"] <= 0 or a["article_number"] > settings.MAX_CIVIL_CODE_ARTICLE_NUMBER 
    ]
    print(f"{len(flagged)} article(s) flagged for manual review "
          f"(missing a language, or a number outside 1-{settings.MAX_CIVIL_CODE_ARTICLE_NUMBER }):")
    
    for a in flagged[:30]:
        print(f"  - Article {a['article_number']} (page {a['source_page']}): "
              f"ar={'yes' if a['text_ar'] else 'no'}, en={'yes' if a['text_en'] else 'no'}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_articles, f, ensure_ascii=False, indent=2)