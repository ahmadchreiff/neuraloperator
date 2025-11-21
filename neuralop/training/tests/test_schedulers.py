from neuralop.training.schedulers import PhysicsWeightScheduler


def test_none_mode_constant_weight():
    scheduler = PhysicsWeightScheduler(
        initial_weight=0.25, max_weight=1.5, warmup_epochs=10, mode="none"
    )
    assert scheduler.step(epoch=0) == 0.25
    assert scheduler.step(epoch=5) == 0.25


def test_linear_warmup_reaches_max():
    scheduler = PhysicsWeightScheduler(
        initial_weight=0.1, max_weight=1.0, warmup_epochs=4, mode="linear_warmup"
    )
    assert abs(scheduler.step(epoch=0) - 0.1) < 1e-6
    assert abs(scheduler.step(epoch=2) - 0.55) < 1e-6
    assert abs(scheduler.step(epoch=4) - 1.0) < 1e-6
    assert abs(scheduler.step(epoch=10) - 1.0) < 1e-6


def test_plateau_mode_triggers_max():
    scheduler = PhysicsWeightScheduler(
        initial_weight=0.0,
        max_weight=1.0,
        warmup_epochs=3,
        mode="plateau",
        plateau_patience=3,
        plateau_threshold=1e-5,
    )
    history = {"data_loss": [1.0, 0.9, 0.85]}
    # Before patience filled, follow warmup curve
    assert scheduler.step(epoch=1, history=history) < 1.0
    # Plateau detected once patience is met
    history["data_loss"].extend([0.85, 0.8500001, 0.8499])
    assert scheduler.step(epoch=5, history=history) == 1.0
