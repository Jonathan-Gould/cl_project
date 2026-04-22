from typing import Annotated, override

from cl.app import BaseApplication, BaseApplicationConfig, OutputType, RunSummary
from cl.app.model import DurationSeconds
from pydantic import Field

import numpy as np
import cl
from common_cl_code.stimulus_generation import register_stim_plan, make_sequence
import common_cl_code




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

        spatial_footprints = make_sequence()

        with cl.open() as neurons:
            stim_plan = neurons.create_stim_plan()
            register_stim_plan(stim_plan, spatial_footprints, frame_time_s=1 / fps)

            recording = neurons.record(stop_after_seconds=10)
            stim_plan.run()

            if common_cl_code.running_on_real_device:
                recording.wait_until_stopped()
            else:
                for tick in neurons.loop(ticks_per_second=10, stop_after_seconds=10):
                    pass
                recording.stop()

        return RunSummary(
            type    = OutputType.TEXT,
            content = "Application completed successfully.",
        )

    @staticmethod
    @override
    def config_class() -> type[MyApplicationConfig]:
        return MyApplicationConfig
