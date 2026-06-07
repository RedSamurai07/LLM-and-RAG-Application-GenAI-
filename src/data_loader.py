import logging
from typing import List, Dict, Tuple
from src.config import CONFIG

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None  # type: ignore


def load_markrai_dataset() -> Tuple[List[Dict], List[Dict]]:
    """Loads the 8.5k research corpus nodes and 520 QA test cases."""
    logger.info("Fetching corpus and QA subsets from Hugging Face...")
    corpus_ds = load_dataset(CONFIG["dataset_name"], "corpus", split="train")
    qa_ds = load_dataset(CONFIG["dataset_name"], "qa", split="train")

    docs = [{"content": r["contents"], "id": r["doc_id"]} for r in corpus_ds]
    test_set = [
        {"question": r["query"], "ground_truth": r["generation_gt"][0]}
        for r in qa_ds
    ]
    return docs, test_set
