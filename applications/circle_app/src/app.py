from typing import Annotated, override

from cl.app import BaseApplication, BaseApplicationConfig, OutputType, RunSummary
from cl.app.model import DurationSeconds
from pydantic import Field

import numpy as np
import cl
from common_cl_code.stimulus_generation import register_stim_plan, make_sequence
import time


class MyApplicationConfig(BaseApplicationConfig):
    """Configuration for MyApplication."""

    # Example configuration field - replace with your own
    duration: Annotated[
        DurationSeconds,
        Field(
            title="Duration",
            description="Duration of the application run in seconds.",
            default=60,
        ),
    ]

    n_trials: Annotated[
        int,
        Field(
            title="Number of trials",
            description="Number of trials to run.",
            default=1000,
        ),
    ]

    inter_trial_period_s: Annotated[
        DurationSeconds,
        Field(
            title="Inter trial period",
            description="Duration of the inter-trial period in seconds.",
            default=60,
        ),
    ]

    @override
    def estimate_duration_s(self) -> float:
        return self.duration


class MyApplication(BaseApplication[MyApplicationConfig]):
    """My custom application."""

    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng(0)

    @override
    def run(self, config: MyApplicationConfig, output_directory: str) -> RunSummary | None:
        fps = 25

        with cl.open() as neurons:
            recording = neurons.record(include_raw_samples=False)

            try:
                for _ in range(config.n_trials):
                    stim_plan = neurons.create_stim_plan()
                    sequence_parameters, stim_parameters = self.random_trial_parameters()
                    sequence = make_sequence(fps=fps, rng=self.rng, **sequence_parameters)
                    register_stim_plan(stim_plan, sequence, frame_time_s=1 / fps, **stim_parameters)
                    stim_plan.run()
                    time.sleep(config.inter_trial_period_s)
            except Exception as e:
                raise e
            finally:
                recording.stop()

        return RunSummary(
            type    = OutputType.TEXT,
            content = "Application completed successfully.",
        )

    def random_trial_parameters(self):
        sequence_parameters = dict(
            seconds_per_rotation = self.rng.choice([1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32]),
            rotations= 5 + self.rng.choice(np.linspace(0,1,9)[:-1]),
            shuffle_axes = self.rng.choice([{}, {0}, {1}, {0,1}], p=[.9,1/30, 1/30, 1/30])
        )

        stim_parameters = dict(
            max_amplitude=self.rng.choice([0.5, 1, 1.5,2, 2.5, 3]),
            phase_width_us=self.rng.choice([80, 160, 320, 640, 1280]),
            burst_hz=self.rng.choice([40, 60, 80, 100, 120, 140, 160, 180, 200]),
        )
        return sequence_parameters, stim_parameters


    @staticmethod
    @override
    def config_class() -> type[MyApplicationConfig]:
        return MyApplicationConfig
