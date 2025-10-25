import os

import pytest

from paraphrase_gen.inference.generate import Paraphraser


@pytest.mark.skipif(
    not os.path.exists("runs/t5_small_qqp"), reason="checkpoint missing"
)
def test_generate_smokef():
    p = Paraphraser("runs/t5_small_qqp", max_new_tokens=16, num_beams=2)
    out = p.generate(["The meeting is at ten tomorrow."])
    assert isinstance(out, list) and len(out) == 1
    assert isinstance(out[0], str) and len(out[0]) > 0
