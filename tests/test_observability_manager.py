from dataclasses import dataclass

from ardagen.observability.manager import ObservabilityManager
from ardagen.observability.events import ObservabilityEventType


@dataclass
class DummyModel:
    value: int

    def model_dump(self):
        return {"value": self.value}


def test_observability_manager_records_events_and_emits_trace():
    captured = []

    def emitter(agent_name, stage, event_type, payload):
        captured.append((agent_name, stage, event_type, payload))

    manager = ObservabilityManager(trace_emitter=emitter)
    manager.stage_started("spec", 1)
    manager.stage_completed("spec", DummyModel(42))
    manager.stage_failed("spec", "boom")

    assert [event.event_type for event in manager.events] == [
        ObservabilityEventType.STAGE_STARTED,
        ObservabilityEventType.STAGE_COMPLETED,
        ObservabilityEventType.STAGE_FAILED,
    ]
    assert len(captured) == 3
    assert captured[0][2] == "stage_started"
    assert "\"value\": 42" in captured[1][3]
