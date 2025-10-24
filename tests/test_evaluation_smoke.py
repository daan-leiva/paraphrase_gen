import os
from pathlib import Path
from subprocess import CalledProcessError, run

import pytest
from huggingface_hub.errors import HfHubHTTPError


@pytest.mark.skipif(
    not os.path.exists("runs/t5_small_qqp"), reason="checkpoint missing"
)
@pytest.mark.network
def test_evaluate_script_creates_outputs(tmp_path: Path):
    out_dir = tmp_path / "eval_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "paraphrase_gen.evaluation.evaluate",
        "--model_path",
        "runs/t5_small_qqp",
        "--dataset_name",
        "glue",
        "--dataset_config",
        "qqp",
        "--num_samples",
        "20",
        "--out_dir",
        str(out_dir),
    ]
    try:
        run(cmd, check=True, capture_output=True, text=True)
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "samples.csv").exists()
    except CalledProcessError as e:
        raise AssertionError(f"Evaluation script failed: {e.stderr}") from e
    except HfHubHTTPError as e:
        pytest.skip(f"Hugging Face Hub unavailable: {e}")
