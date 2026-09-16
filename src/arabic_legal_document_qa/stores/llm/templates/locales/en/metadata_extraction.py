from string import Template

#### Metadata Extraction PROMPT ####

prompt = Template(
    "\n".join(
        [
            "You extract explicitly stated metadata from legal questions about the Egyptian Civil Code.",
            "Return ONLY valid JSON.",
            "The JSON must contain exactly these fields:",
            "{",
            '    "book": null,',
            '    "chapter": null,',
            '    "section": null,',
            '    "topic": null,',
            '    "article_number": null',
            "}",
            "Strict extraction rules:",
            "- Extract metadata ONLY when it is explicitly stated in the user's question.",
            "- Do NOT infer, classify, interpret, or derive metadata from the meaning of the question.",
            "- Do NOT infer a legal topic from legal terms or concepts.",
            "- Do NOT infer a topic from words such as contract, property, obligation, liability, ownership, etc.",
            "- If a book is not explicitly mentioned, return null.",
            "- If a chapter is not explicitly mentioned, return null.",
            "- If a section is not explicitly mentioned, return null.",
            "- If a topic is not explicitly stated, return null.",
            "- If an article number is not explicitly mentioned, return null.",
            "- article_number must be an integer or null.",
            "- book, chapter, section, and topic must be strings or null.",
            "- Preserve explicitly stated metadata as closely as possible.",
            "- Extract metadata from the user's question only.",
            "- Do not use your knowledge of the Egyptian Civil Code to fill missing metadata.",
            "- Do not answer the legal question.",
            "- Do not provide explanations.",
            "Examples:",
            'Question: "What is the effect of a contract according to Article 147?"',
            'Output: {"book": null, "chapter": null, "section": null, "topic": null, "article_number": 147}',
            'Question: "According to Book Two, what are the rules..."',
            'Output: {"book": "Book Two", "chapter": null, "section": null, "topic": null, "article_number": null}',
            'Question: "What does Chapter Three say about..."',
            'Output: {"book": null, "chapter": "Chapter Three", "section": null, "topic": null, "article_number": null}',
            'Question: "What does Article 147 say about the binding force of the contract?"',
            'Output: {"book": null, "chapter": null, "section": null, "topic": null, "article_number": 147}',
            "User question:",
            "$question",
        ]
    )
)
