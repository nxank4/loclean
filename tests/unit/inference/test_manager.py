from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from semantix.inference.manager import LocalInferenceEngine


@pytest.fixture
def temp_cache_dir(tmp_path):
    return tmp_path / "test_cache"


@pytest.fixture
def mock_llama():
    mock_llama_instance = Mock()
    mock_llama_instance.create_completion = Mock()
    return mock_llama_instance


@pytest.fixture
def mock_llama_class(mock_llama):
    with patch("semantix.inference.manager.Llama", return_value=mock_llama):
        yield mock_llama


@pytest.fixture
def mock_grammar():
    mock_grammar_instance = Mock()
    return mock_grammar_instance


@pytest.fixture
def mock_grammar_class(mock_grammar):
    with patch("semantix.inference.manager.LlamaGrammar", return_value=mock_grammar):
        with patch(
            "semantix.inference.manager.LlamaGrammar.from_string", return_value=mock_grammar
        ):
            yield mock_grammar


@pytest.fixture
def mock_cache():
    mock_cache_instance = Mock()
    mock_cache_instance.get_batch = Mock(return_value={})
    mock_cache_instance.set_batch = Mock()
    mock_cache_instance._hash = Mock(
        side_effect=lambda text, instruction: f"hash_{text}_{instruction}"
    )
    return mock_cache_instance


@pytest.fixture
def mock_cache_class(mock_cache):
    with patch("semantix.cache.SemantixCache", return_value=mock_cache):
        yield mock_cache


@pytest.fixture
def mock_model_path(temp_cache_dir):
    temp_cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = temp_cache_dir / "test_model.gguf"
    model_path.touch()
    return model_path


@pytest.fixture
def mock_hf_download(mock_model_path):
    with patch("semantix.inference.manager.hf_hub_download", return_value=str(mock_model_path)):
        yield


def test_init_with_custom_cache_dir(
    temp_cache_dir,
    mock_model_path,
    mock_llama_class,
    mock_grammar_class,
    mock_cache_class,
    mock_hf_download,
):
    engine = LocalInferenceEngine(cache_dir=temp_cache_dir)

    assert engine.cache_dir == temp_cache_dir
    assert engine.model_path == mock_model_path
    assert engine.llm is not None
    assert engine.grammar is not None
    assert engine.cache is not None


def test_init_with_default_cache_dir(
    mock_llama_class, mock_grammar_class, mock_cache_class, mock_hf_download
):
    with patch("semantix.inference.manager.Path.home", return_value=Path("/home/test")):
        with patch("pathlib.Path.mkdir"):
            expected_cache_dir = Path("/home/test/.cache/semantix")
            mock_model_path = expected_cache_dir / LocalInferenceEngine.MODEL_FILENAME

            def mock_exists(*args):
                path_self = args[0] if args else None
                return str(path_self) == str(mock_model_path) if path_self else False

            with patch("pathlib.Path.exists", side_effect=mock_exists):
                engine = LocalInferenceEngine()

                assert engine.cache_dir == expected_cache_dir


def test_get_model_path_existing_file(temp_cache_dir, mock_model_path, mock_hf_download):
    with patch("semantix.inference.manager.Llama"):
        with patch("semantix.inference.manager.LlamaGrammar"):
            with patch("semantix.cache.SemantixCache"):
                engine = LocalInferenceEngine(cache_dir=temp_cache_dir)
                path = engine._get_model_path()

                assert path == mock_model_path


def test_get_model_path_downloads_when_missing(temp_cache_dir):
    with patch("semantix.inference.manager.Path.exists", return_value=False):
        with patch("semantix.inference.manager.Llama"):
            with patch("semantix.inference.manager.LlamaGrammar"):
                with patch("semantix.cache.SemantixCache"):
                    with patch("semantix.inference.manager.hf_hub_download") as mock_download:
                        mock_download.return_value = str(temp_cache_dir / "downloaded_model.gguf")

                        engine = LocalInferenceEngine(cache_dir=temp_cache_dir)
                        mock_download.reset_mock()
                        engine._get_model_path()

                        mock_download.assert_called_once_with(
                            repo_id=LocalInferenceEngine.MODEL_REPO,
                            filename=LocalInferenceEngine.MODEL_FILENAME,
                            local_dir=temp_cache_dir,
                        )


def test_get_json_grammar(
    temp_cache_dir, mock_model_path, mock_llama_class, mock_cache_class, mock_hf_download
):
    with patch("semantix.inference.manager.LlamaGrammar") as mock_grammar_class:
        mock_grammar_instance = Mock()
        mock_grammar_class.from_string = Mock(return_value=mock_grammar_instance)

        engine = LocalInferenceEngine(cache_dir=temp_cache_dir)
        mock_grammar_class.from_string.reset_mock()
        grammar = engine._get_json_grammar()

        assert grammar == mock_grammar_instance
        mock_grammar_class.from_string.assert_called_once()


def test_clean_batch_all_cached(
    mock_llama_class,
    mock_grammar_class,
    mock_cache_class,
    mock_hf_download,
    temp_cache_dir,
    mock_model_path,
):
    cached_results = {
        "10kg": {"reasoning": "test", "value": 10.0, "unit": "kg"},
        "500g": {"reasoning": "test", "value": 500.0, "unit": "g"},
    }

    mock_cache_class.get_batch = Mock(return_value=cached_results)

    engine = LocalInferenceEngine(cache_dir=temp_cache_dir)
    result = engine.clean_batch(["10kg", "500g"], "Extract weight")

    assert result == cached_results
    assert not engine.llm.create_completion.called
    assert not mock_cache_class.set_batch.called


def test_clean_batch_partial_cache(
    mock_llama_class,
    mock_grammar_class,
    mock_cache_class,
    mock_hf_download,
    temp_cache_dir,
    mock_model_path,
):
    cached_results = {"10kg": {"reasoning": "test", "value": 10.0, "unit": "kg"}}
    mock_cache_class.get_batch = Mock(return_value=cached_results)

    llm_output = {"choices": [{"text": '{"reasoning": "test", "value": 500.0, "unit": "g"}'}]}
    mock_llama_class.create_completion = Mock(return_value=llm_output)

    engine = LocalInferenceEngine(cache_dir=temp_cache_dir)
    result = engine.clean_batch(["10kg", "500g"], "Extract weight")

    assert "10kg" in result
    assert "500g" in result
    assert result["10kg"] == cached_results["10kg"]
    assert result["500g"]["value"] == 500.0
    assert result["500g"]["unit"] == "g"
    mock_cache_class.set_batch.assert_called_once()


def test_clean_batch_successful_inference(
    mock_llama_class,
    mock_grammar_class,
    mock_cache_class,
    mock_hf_download,
    temp_cache_dir,
    mock_model_path,
):
    llm_output = {
        "choices": [{"text": '{"reasoning": "Extracted weight", "value": 10.0, "unit": "kg"}'}]
    }
    mock_llama_class.create_completion = Mock(return_value=llm_output)

    engine = LocalInferenceEngine(cache_dir=temp_cache_dir)
    result = engine.clean_batch(["10kg"], "Extract weight")

    assert "10kg" in result
    assert result["10kg"]["value"] == 10.0
    assert result["10kg"]["unit"] == "kg"
    assert result["10kg"]["reasoning"] == "Extracted weight"
    mock_cache_class.set_batch.assert_called_once()


def test_clean_batch_json_decode_error(
    mock_llama_class,
    mock_grammar_class,
    mock_cache_class,
    mock_hf_download,
    temp_cache_dir,
    mock_model_path,
):
    llm_output = {"choices": [{"text": "invalid json {"}]}
    mock_llama_class.create_completion = Mock(return_value=llm_output)

    engine = LocalInferenceEngine(cache_dir=temp_cache_dir)
    result = engine.clean_batch(["10kg"], "Extract weight")

    assert "10kg" in result
    assert result["10kg"] is None
    assert not mock_cache_class.set_batch.called


def test_clean_batch_missing_keys(
    mock_llama_class,
    mock_grammar_class,
    mock_cache_class,
    mock_hf_download,
    temp_cache_dir,
    mock_model_path,
):
    llm_output = {"choices": [{"text": '{"value": 10.0}'}]}
    mock_llama_class.create_completion = Mock(return_value=llm_output)

    engine = LocalInferenceEngine(cache_dir=temp_cache_dir)
    result = engine.clean_batch(["10kg"], "Extract weight")

    assert "10kg" in result
    assert result["10kg"] is None
    assert not mock_cache_class.set_batch.called


def test_clean_batch_inference_exception(
    mock_llama_class,
    mock_grammar_class,
    mock_cache_class,
    mock_hf_download,
    temp_cache_dir,
    mock_model_path,
):
    mock_llama_class.create_completion = Mock(side_effect=Exception("LLM error"))

    engine = LocalInferenceEngine(cache_dir=temp_cache_dir)
    result = engine.clean_batch(["10kg"], "Extract weight")

    assert "10kg" in result
    assert result["10kg"] is None
    assert not mock_cache_class.set_batch.called


def test_clean_batch_multiple_items_with_mixed_results(
    mock_llama_class,
    mock_grammar_class,
    mock_cache_class,
    mock_hf_download,
    temp_cache_dir,
    mock_model_path,
):
    cached_results = {"item1": {"reasoning": "cached", "value": 1.0, "unit": "kg"}}
    mock_cache_class.get_batch = Mock(return_value=cached_results)

    def create_completion_side_effect(*args, **kwargs):
        prompt = kwargs.get("prompt", "")
        if "item2" in prompt:
            return {"choices": [{"text": '{"reasoning": "inferred", "value": 2.0, "unit": "g"}'}]}
        elif "item3" in prompt:
            return {"choices": [{"text": "invalid json"}]}
        return {"choices": [{"text": '{"reasoning": "test", "value": 0.0, "unit": "kg"}'}]}

    mock_llama_class.create_completion = Mock(side_effect=create_completion_side_effect)

    engine = LocalInferenceEngine(cache_dir=temp_cache_dir)
    result = engine.clean_batch(["item1", "item2", "item3"], "Extract weight")

    assert result["item1"]["value"] == 1.0
    assert result["item2"]["value"] == 2.0
    assert result["item3"] is None
    assert mock_llama_class.create_completion.call_count == 2


def test_clean_batch_only_caches_valid_results(
    mock_llama_class,
    mock_grammar_class,
    mock_cache_class,
    mock_hf_download,
    temp_cache_dir,
    mock_model_path,
):
    call_order = []

    def create_completion_side_effect(*args, **kwargs):
        prompt = kwargs.get("prompt", "")
        call_order.append(prompt)
        if '"valid_item"' in prompt:
            return {"choices": [{"text": '{"reasoning": "ok", "value": 10.0, "unit": "kg"}'}]}
        return {"choices": [{"text": "invalid json {"}]}

    mock_llama_class.create_completion = Mock(side_effect=create_completion_side_effect)
    mock_cache_class.get_batch = Mock(return_value={})
    mock_cache_class.set_batch.reset_mock()

    engine = LocalInferenceEngine(cache_dir=temp_cache_dir)

    assert engine.cache is mock_cache_class, "engine.cache should be the mocked instance"

    result = engine.clean_batch(["valid_item", "invalid_item"], "Extract weight")

    assert engine.cache.set_batch.called, "set_batch should be called"

    call_args = engine.cache.set_batch.call_args
    assert call_args is not None, "set_batch should have been called with arguments"

    cached_items = call_args[0][0]
    cached_results = call_args[0][2]

    assert "valid_item" in cached_items, f"valid_item should be in cached items: {cached_items}"
    assert "invalid_item" not in cached_items, (
        f"invalid_item should not be in cached items: {cached_items}"
    )
    assert "valid_item" in cached_results, (
        f"valid_item should be in cached results: {list(cached_results.keys())}"
    )
    assert "invalid_item" not in cached_results, (
        f"invalid_item should not be in cached results: {list(cached_results.keys())}"
    )

    assert result["valid_item"] is not None
    assert result["invalid_item"] is None
