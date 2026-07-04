import json

import pytest

from sciml.problems.epidemiology.config import EpiConfig
from sciml.problems.swe.config import SWEConfig, TrainConfig
from sciml.problems.wave_obstacle.config import WaveObstacleConfig


def test_swe_defaults():
    c = SWEConfig()
    assert c.model.n_sensors == 100 and c.data.grid == 500 and c.train.n_iter == 40000


def test_swe_roundtrip():
    c = SWEConfig()
    assert SWEConfig.from_dict(c.to_dict()).to_dict() == c.to_dict()


def test_nested_from_dict():
    c = SWEConfig.from_dict({"name": "x", "train": {"n_iter": 5, "batch": 2}})
    assert isinstance(c.train, TrainConfig) and c.train.n_iter == 5
    assert c.model.n_sensors == 100  # untouched defaults preserved


def test_unknown_key_raises():
    with pytest.raises(TypeError):
        SWEConfig.from_dict({"nope": 1})


def test_json_roundtrip(tmp_path):
    for cls in (SWEConfig, EpiConfig, WaveObstacleConfig):
        c = cls()
        p = tmp_path / f"{cls.__name__}.json"
        c.save(p)
        json.loads(p.read_text())
        assert cls.load(p).to_dict() == c.to_dict()


def test_epi_and_wave_defaults():
    assert EpiConfig().model.model == "SIRS"
    assert WaveObstacleConfig().params.b == 0.3
    assert len(WaveObstacleConfig().train.phases) == 5
