from train import get_lr_multiplier


def test_warmup_learning_rate_increases():
    first = get_lr_multiplier(1, 4000)
    middle = get_lr_multiplier(2000, 4000)
    peak = get_lr_multiplier(4000, 4000)

    assert first < middle < peak
    assert peak == 1.0


def test_learning_rate_decays_after_warmup():
    peak = get_lr_multiplier(4000, 4000)
    after = get_lr_multiplier(8000, 4000)

    assert after < peak


def test_zero_warmup_is_constant():
    assert get_lr_multiplier(1, 0) == 1.0
    assert get_lr_multiplier(100, 0) == 1.0
