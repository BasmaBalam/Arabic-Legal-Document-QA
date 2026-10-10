import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness

from arabic_legal_document_qa.configs.config import get_settings
from arabic_legal_document_qa.controllers.RAGController import RAGController
from arabic_legal_document_qa.stores.llm.LLMProviderFactory import LLMProviderFactory
from arabic_legal_document_qa.stores.llm.templates.template_parser import TemplateParser
from arabic_legal_document_qa.stores.vectordb.VectorDBProviderFactory import (
    VectorDBProviderFactory,
)

# 20-Question Legal Test Set
EVAL_QUESTIONS_20 = [
    # --- Arabic Legal Questions ---
    "ما هي المصادر التي يلجأ إليها القاضي عند عدم وجود نص تشريعي؟",
    "هل يجوز إلغاء نص تشريعي بتشريع لاحق؟ وما هي الشروط؟",
    "متى يعتبر استعمال الحق غير مشروع وفقاً للمادة 5؟",
    "ما هو القانون الواجب التطبيق على الأشخاص الاعتبارية الأجنبية؟",
    "ما هي سن الرشد القانونية لمباشرة التصرفات المدنية؟",
    "ما هو التعريف القانوني للعقار بالتخصيص؟",
    "كيف يتم انعقاد العقد بين الطرفين؟",
    "ما هي الآثار القانونية المترتبة على دفع العربون؟",
    "ما هو حكم التصرفات المالية التي يصدرها الصبي المميز؟",
    "متى يعتبر الغلط جوهرياً بحيث يجيز طلب إبطال العقد؟",
    # --- English Legal Questions ---
    "What are the primary sources of law under Egyptian Civil Law when there is no applicable legislative provision?",
    "Under what conditions is the exercise of a right considered unlawful under the Egyptian Civil Code?",
    "How is the legal capacity and applicable law determined for foreign juristic persons operating in Egypt?",
    "Which law governs contractual obligations when the contracting parties do not share a common domicile?",
    "At what age does a person reach legal majority under Egyptian Civil Law?",
    "What constitutes an immovable by destination under the classification of property?",
    "How is a contract formed through offer and acceptance under Article 89?",
    "What are the legal consequences of paying earnest money upon contract execution?",
    "What are the legal effects of financial transactions entered into by a discerning minor?",
    "When is a mistake considered essential to warrant the avoidance of a contract?",
]


@pytest.fixture(scope="module")
def rag_controller():
    settings = get_settings()
    llm_factory = LLMProviderFactory(settings)
    vector_factory = VectorDBProviderFactory(settings)

    generation_client = llm_factory.create(provider=settings.GENERATION_BACKEND)
    generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    embedding_client = llm_factory.create(provider=settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE,
    )

    vectordb_client = vector_factory.create(provider=settings.VECTOR_DB_BACKEND)
    vectordb_client.connect()

    template_parser = TemplateParser(language=settings.DEFAULT_LANG)

    return RAGController(
        vectordb_client=vectordb_client,
        generation_client=generation_client,
        embedding_client=embedding_client,
        template_parser=template_parser,
    )


def test_ragas_faithfulness_gate(rag_controller):
    """Quality Gate: Evaluates 20 legal questions and asserts faithfulness >= 0.75."""
    questions, answers, contexts = [], [], []

    for q in EVAL_QUESTIONS_20:
        result = rag_controller.answer(question=q, limit=3)
        if result is None:
            continue

        answer, sources, retrieved_contexts = result
        questions.append(q)
        answers.append(answer)
        contexts.append(retrieved_contexts)

    assert (
        len(questions) == 20
    ), f"Expected 20 evaluated questions, got {len(questions)}"

    eval_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
    )

    eval_results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness],
    )

    faithfulness_score = eval_results["faithfulness"]
    print(f"\n[CI Quality Gate] RAGAS Faithfulness Score: {faithfulness_score:.4f}")

    # Quality Gate condition: Pipeline fails if faithfulness < 0.75
    assert faithfulness_score >= 0.75, (
        f"CI Pipeline Failed: RAGAS Faithfulness score ({faithfulness_score:.4f}) "
        f"is below required threshold of 0.75."
    )
