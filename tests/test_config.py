from src.config import CONFIG


def test_config_has_required_keys():
    assert "embedding_model" in CONFIG
    assert "dataset_name" in CONFIG
    assert "dense_weight" in CONFIG
    assert "top_k" in CONFIG


def test_dense_weight_is_valid():
    assert 0 < CONFIG["dense_weight"] < 1


def test_top_k_is_positive():
    assert CONFIG["top_k"] > 0


def test_embedding_model_is_string():
    assert isinstance(CONFIG["embedding_model"], str)


def test_dataset_name_is_string():
    assert isinstance(CONFIG["dataset_name"], str)
