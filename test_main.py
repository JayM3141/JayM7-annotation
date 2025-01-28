def test_imports():
    import torch
    import transformers
    assert torch.__version__
    assert transformers.__version__
