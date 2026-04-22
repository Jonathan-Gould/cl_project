from typing import Annotated, override

from cl.app import BaseApplication, BaseApplicationConfig, OutputType, RunSummary
from cl.app.model import DurationSeconds
from pydantic import Field


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

    @override
    def run(self, config: MyApplicationConfig, output_directory: str) -> RunSummary | None:
        # TODO: Implement your application logic here
        return RunSummary(
            type    = OutputType.TEXT,
            content = "Application completed successfully.",
        )

    @staticmethod
    @override
    def config_class() -> type[MyApplicationConfig]:
        return MyApplicationConfig
