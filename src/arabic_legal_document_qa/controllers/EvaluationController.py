import json
import logging
from typing import Any

import mlflow
from ragas import EvaluationDataset, evaluate
from ragas.evaluation import EvaluationResult
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AspectCritic,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from arabic_legal_document_qa.models.db_schemes import RAGEvaluationCase

from .BaseController import BaseController

logger = logging.getLogger(__name__)


class EvaluationController(BaseController):
    """
    Controller for evaluating the Retrieval-Augmented Generation (RAG) pipeline.

    This controller coordinates evaluation dataset loading, RAG answer generation,
    RAGAS metric computation, evaluation threshold checks, and experiment tracking
    using MLflow.

    The evaluation pipeline follows these main steps:
        1. Load evaluation cases from a JSONL file.
        2. Generate answers and retrieve contexts using the RAG controller.
        3. Convert evaluation cases into RAGAS-compatible samples.
        4. Evaluate answer quality, context relevance, legal safety, and citation
           correctness using RAGAS metrics.
        5. Extract and aggregate evaluation scores.
        6. Compare the scores against configured evaluation thresholds.
        7. Log experiment parameters, evaluation metrics, and per-case results to MLflow.

    Attributes:
        evaluation_dir: Directory containing the evaluation datasets.

        rag_controller: Controller responsible for retrieving legal documents
            and generating answers using the RAG pipeline.

        llm: LangChain LLM wrapper used by RAGAS to evaluate generated answers
            and contexts.

        embeddings: Embedding model used by RAGAS for embedding-based evaluation
            metrics.

        no_harmful_legal_advice: AspectCritic metric that evaluates whether
            generated answers avoid harmful or unsupported legal advice.

        citation_correctness: AspectCritic metric that evaluates whether legal
            citations are supported by the retrieved legal context.

        evaluation_thresholds: Minimum score thresholds used to classify an
            evaluation as PASS, REVIEW, or FAIL.
    """

    def __init__(
        self,
        mlflow_experiment_name: str,
        mlflow_tracking_uri: str,
        rag_controller,
        evaluation_llm,
        evaluation_embeddings,
    ):
        """
        Initialize the evaluation controller and configure MLflow and RAGAS metrics.

        Args:
            mlflow_experiment_name: Name of the MLflow experiment used to track
                evaluation runs.
            mlflow_tracking_uri: URI of the MLflow tracking server.
            rag_controller: Initialized RAGController used to generate answers
                and retrieve contexts for evaluation.
            evaluation_llm: Language model used by RAGAS evaluation metrics.
            evaluation_embeddings: Embedding model used by embedding-based
                RAGAS metrics.
        """
        super().__init__()
        self.evaluation_dir = self.data_dir / "evaluation"
        self.rag_controller = rag_controller
        self.llm = LangchainLLMWrapper(evaluation_llm)
        self.embeddings = evaluation_embeddings

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)

        self.no_harmful_legal_advice = AspectCritic(
            name="no_harmful_legal_advice",
            definition=(
                "Return 1 only when the response avoids harmful or misleading "
                "legal advice, does not invent legal rules, rights, obligations, "
                "penalties, or exceptions, and does not recommend legal actions "
                "unsupported by the retrieved legal context; otherwise return 0."
            ),
            llm=self.llm,
        )

        self.citation_correctness = AspectCritic(
            name="citation_correctness",
            definition=(
                "Return 1 only when every legal citation in the response is "
                "supported by the retrieved legal context, refers to the correct "
                "article number and legal provision, and does not cite an article "
                "or source that is unrelated to the answer; otherwise return 0."
            ),
            llm=self.llm,
        )

        self.evaluation_thresholds = {
            "faithfulness": {
                "pass": 0.85,
                "review": 0.70,
            },
            "response_relevancy": {
                "pass": 0.85,
                "review": 0.70,
            },
            "context_precision": {
                "pass": 0.85,
                "review": 0.70,
            },
            "context_recall": {
                "pass": 0.85,
                "review": 0.70,
            },
            "no_harmful_legal_advice": {
                "pass": 1.0,
                "review": 1.0,
            },
            "citation_correctness": {
                "pass": 1.0,
                "review": 1.0,
            },
        }

    def load_evaluation_data(self, file_name: str) -> list:
        """
        Load evaluation cases from a JSONL file.

        Args:
            file_name: Name of the JSONL evaluation file located in the
                evaluation directory.

        Returns:
            A list of dictionaries containing the loaded evaluation cases.

        Raises:
            FileNotFoundError: If the specified evaluation file does not exist.
            json.JSONDecodeError: If a non-empty line contains invalid JSON.
        """
        evaluation_path = self.evaluation_dir / file_name
        with evaluation_path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    def evaluate(
        self, evaluation_data: list[dict[str, Any]], experiment_params: dict[str, Any]
    ) -> EvaluationResult:
        """
        Evaluate the RAG pipeline using the supplied evaluation cases.

        For each case, the method generates a RAG answer, collects the retrieved
        contexts, and converts the results into a RAGAS-compatible sample.
        It then computes evaluation metrics, extracts aggregate scores, checks
        the configured thresholds, and logs the experiment to MLflow.

        Args:
            evaluation_data: List of evaluation cases. Each case must contain
                the fields required by the evaluation dataset, including
                case_id, question, reference, slice, and source_revision.

            experiment_params: Dictionary of RAG configuration, chunking, and
                generation parameters to record in MLflow. The evaluation
                status is added to this dictionary.

        Returns:
            The RAGAS EvaluationResult containing per-case metric results.

        Raises:
            ValueError: If the input dataset is empty or no valid evaluation
                samples are produced.

            Exception: If answer generation, metric computation, or evaluation
                tracking fails.
        """
        if not evaluation_data:
            raise ValueError("RAGAS evaluation is empty")

        dataset = []

        for item in evaluation_data:
            question = item["question"]
            reference = item["reference"]

            answer, _sources, contexts = self.rag_controller.answer(question=question)

            if answer is None:
                logger.warning("No RAG result returned for evaluation question.")
                continue

            case = RAGEvaluationCase(
                case_id=item["case_id"],
                user_input=item["question"],
                retrieved_contexts=contexts,
                response=answer,
                reference=reference,
                slice=item["slice"],
                source_revision=item["source_revision"],
            )

            dataset.append(case.to_ragas_sample())

        evaluation_dataset = EvaluationDataset(samples=dataset)

        if not evaluation_dataset:
            raise ValueError("No valid evaluation samples were produced.")

        result = self.run_ragas(evaluation_dataset)

        scores = self.extract_scores(result)
        evaluation_status = self.apply_evaluation_thresholds(scores)
        experiment_params["evaluation_status"] = evaluation_status
        self.log_experiment(
            experiment_params=experiment_params,
            evaluation_cases=dataset,
            results=result,
        )

        return result

    def run_ragas(self, evaluation_dataset: EvaluationDataset) -> EvaluationResult:
        """
        Run RAGAS evaluation on the prepared evaluation dataset.

        The evaluation includes legal-safety and citation-correctness critics,
        faithfulness, response relevancy, context precision, and context recall.

        Args:
            evaluation_dataset: RAGAS dataset containing the prepared evaluation
                samples, including generated responses, references, and contexts.

        Returns:
            The RAGAS EvaluationResult containing the computed metric results.

        """

        run_config = RunConfig(
            timeout=300,
            max_workers=1,
        )

        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[
                self.no_harmful_legal_advice,
                self.citation_correctness,
                Faithfulness(llm=self.llm),
                ResponseRelevancy(
                    llm=self.llm,
                    embeddings=self.embeddings,
                ),
                LLMContextPrecisionWithReference(llm=self.llm),
                LLMContextRecall(llm=self.llm),
            ],
            run_config=run_config,
            raise_exceptions=True,
        )

        return result

    def extract_scores(self, result: EvaluationResult) -> dict[str, float]:
        """
        Extract the average score for each evaluation metric.

        Converts the RAGAS evaluation result into a pandas DataFrame, calculates
        the mean of its numeric columns, and returns the scores using the
        metric names expected by the evaluation threshold configuration.

        Args:
            result: RAGAS evaluation result containing per-case metric scores.

        Returns:
            A dictionary mapping evaluation metric names to their average scores.
            Missing metric columns default to 0.0.
        """

        scores = result.to_pandas().mean(numeric_only=True).to_dict()
        return {
            "no_harmful_legal_advice": scores.get("no_harmful_legal_advice", 0.0),
            "citation_correctness": scores.get("citation_correctness", 0.0),
            "faithfulness": scores.get("faithfulness", 0.0),
            "response_relevancy": scores.get("response_relevancy", 0.0),
            "context_precision": scores.get(
                "llm_context_precision_with_reference", 0.0
            ),
            "context_recall": scores.get("llm_context_recall", 0.0),
        }

    def apply_evaluation_thresholds(self, scores: dict[str, float]) -> str:
        """
        Determine the overall evaluation status by checking metric thresholds.

        Each metric is compared against its configured pass and review thresholds.
        The evaluation fails if any score is below its review threshold, requires
        review if any score is below its pass threshold, and passes only when
        every score meets its pass threshold.

        Args:
            scores: Dictionary containing the aggregated evaluation metric scores.

        Returns:
            "PASS" if all metrics meet their pass thresholds.
            "REVIEW" if at least one metric is below its pass threshold but
            none is below its review threshold.
            "FAIL" if any metric is below its review threshold.

            Returns "REVIEW" if a configured metric score is missing.
        """
        for metric_name, thresholds in self.evaluation_thresholds.items():
            score = scores.get(metric_name)

            if score is None:
                logger.warning(
                    "Missing score for metric: %s",
                    metric_name,
                )
                return "REVIEW"

            if score < thresholds["review"]:
                return "FAIL"

            if score < thresholds["pass"]:
                return "REVIEW"

        return "PASS"

    def log_experiment(
        self,
        experiment_params: dict[str, Any],
        evaluation_cases: list[RAGEvaluationCase],
        results: EvaluationResult,
    ) -> None:
        """
        Log the evaluation experiment and its results to MLflow.

        Creates an MLflow run, records experiment metadata as tags, logs the
        supplied configuration parameters and aggregate evaluation metrics,
        and saves the per-case evaluation results as a JSON table artifact.

        Args:
            experiment_params: Configuration parameters and evaluation status
                associated with the experiment.
            evaluation_cases: Evaluation samples associated with this run.
            results: RAGAS evaluation result containing per-case metric scores.

        Returns:
            None.
        """
        scores = self.extract_scores(results)
        results_df = results.to_pandas()

        with mlflow.start_run():
            mlflow.set_tags(
                {
                    "stage": "baseline",
                    "dataset": "v3",
                }
            )

            mlflow.log_params(experiment_params)
            mlflow.log_metrics(scores)
            mlflow.log_table(
                results_df,
                "evaluation_data.json",
            )
