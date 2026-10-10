import random
from locust import HttpUser, task, between

# Arabic and English legal questions based on the Egyptian Civil Code
LEGAL_TEST_QUESTIONS = [
    # Arabic questions
    "ما هي شروط صحة العقد في القانون المدني المصري؟",
    "متى يكون العقد باطلاً بطلاناً مطلقاً؟",
    "ما هي أحكام مسؤولية المتبوع عن أعمال تابعه وفقاً للمادة 174؟",
    "كيف يتحدد القانون الواجب التطبيق عند عدم وجود نص تشريعي؟",
    "ما هي القوة القاهرة وما أثرها على تنفيذ الالتزامات التعاقدية؟",
    "متى يعتبر استعمال الحق غير مشروع؟",
    "ما هي سن الرشد القانونية لمباشرة التصرفات المدنية؟",
    "ما هو حكم التصرفات المالية التي يصدرها الصبي المميز؟",
    "ما هي الآثار القانونية المترتبة على دفع العربون؟",
    "متى يعتبر الغلط جوهرياً بحيث يجيز طلب إبطال العقد؟",
    # English questions
    "What are the requirements for a valid contract under the Egyptian Civil Code?",
    "When is a contract considered absolutely null and void?",
    "What are the rules governing a principal's liability for the acts of their subordinate under Article 174?",
    "How is the applicable law determined when there is no legislative provision?",
    "What is force majeure, and how does it affect the performance of contractual obligations?",
    "When is the exercise of a right considered unlawful?",
    "What is the legal age of majority for conducting civil transactions?",
    "What rules apply to financial transactions made by a discerning minor?",
    "What are the legal effects of paying earnest money?",
    "When is a mistake considered fundamental enough to justify rescinding a contract?",
]


class LegalQARAGUser(HttpUser):
    # Simulate a user waiting 1–3 seconds between tasks.
    wait_time = between(1, 3)

    @task(1)
    def health_check(self):
        """Check API health and the indexed document count."""
        with self.client.get(
            "/api/v1/health",
            catch_response=True,
            name="GET /api/v1/health",
        ) as response:
            try:
                data = response.json()

                health_data = data.get("health_response", {})

                if response.status_code == 200 and "documents_indexed" in health_data:
                    response.success()
                else:
                    response.failure(
                        "Health check failed: unexpected status or response structure"
                    )

            except (ValueError, AttributeError):
                response.failure("Health check returned invalid JSON")

    @task(4)
    def ask_blocking_question(self):
        """Test the blocking JSON /ask endpoint."""
        payload = {
            "question": random.choice(LEGAL_TEST_QUESTIONS),
            "limit": 3,
        }

        with self.client.post(
            "/api/v1/ask",
            json=payload,
            catch_response=True,
            name="POST /api/v1/ask [Blocking]",
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text}")
                return

            try:
                data = response.json()

                if isinstance(data, dict) and "answer" in data and "sources" in data:
                    response.success()
                else:
                    response.failure("Missing 'answer' or 'sources' in response JSON")

            except ValueError:
                response.failure("API returned invalid JSON")

    @task(1)
    def ask_empty_question_validation(self):
        """Check that an empty question is rejected with HTTP 422."""
        payload = {
            "question": "",
            "limit": 3,
        }

        with self.client.post(
            "/api/v1/ask",
            json=payload,
            catch_response=True,
            name="POST /api/v1/ask [Validation 422]",
        ) as response:
            if response.status_code == 422:
                response.success()
            else:
                response.failure(
                    f"Expected HTTP 422 for empty question, "
                    f"got {response.status_code}"
                )
