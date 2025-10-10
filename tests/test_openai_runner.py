import pytest

from ardagen.agents.openai_runner import OpenAIAgentRunner


class _Stub:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _make_runner() -> OpenAIAgentRunner:
    from ardagen.agents.json_parser import ResponseJSONParser
    runner = OpenAIAgentRunner.__new__(OpenAIAgentRunner)
    # Manually initialize the components needed for testing
    runner._json_parser = ResponseJSONParser(
        OpenAIAgentRunner._REQ_KEYS,
        OpenAIAgentRunner._EXPECTS_OBJECT
    )
    return runner


def test_extract_payload_from_output_json():
    runner = _make_runner()
    response = _Stub(output=[_Stub(output_json={"foo": 1})])
    payload = runner._json_parser.extract_response_payload(response, "rtl")
    assert payload == {"foo": 1}


def test_extract_payload_from_content_json_dict():
    runner = _make_runner()
    response = _Stub(output=[_Stub(content=[{"json": {"bar": 2}}])])
    payload = runner._json_parser.extract_response_payload(response, "quant")
    assert payload == {"bar": 2}


def test_attribute_json_on_block_is_ignored():
    runner = _make_runner()

    class Block:
        def __init__(self):
            self.content = [{"json": {"keep": True}}]
        def json(self):  # should be ignored
            return {"id": "should_not_use"}

    payload = runner._json_parser.extract_response_payload(_Stub(output=[Block()]), "spec")
    assert payload == {"keep": True}


def test_extract_payload_from_callable_json_accessor():
    runner = _make_runner()

    class WithCallable:
        def __init__(self):
            self.json = lambda: {"callable": True}
            self.type = "output_json"

    response = _Stub(output=[_Stub(content=[WithCallable()])])
    payload = runner._json_parser.extract_response_payload(response, "microarch")
    assert payload == {"callable": True}


def test_extract_payload_from_json_string_text():
    runner = _make_runner()
    response = _Stub(output=[_Stub(content=[_Stub(text='{"baz": 3}')])])
    payload = runner._json_parser.extract_response_payload(response, "spec")
    assert payload == {"baz": 3}


def test_extract_payload_from_top_level_output_text():
    runner = _make_runner()
    response = _Stub(output_text='{"top": 7}')
    payload = runner._json_parser.extract_response_payload(response, "spec")
    assert payload == {"top": 7}


def test_extract_payload_from_top_level_output_parsed_dict():
    runner = _make_runner()
    response = _Stub(output_parsed={"z": 9})
    payload = runner._json_parser.extract_response_payload(response, "rtl")
    assert payload == {"z": 9}


def test_extract_payload_from_nested_json_schema_parsed():
    runner = _make_runner()
    response = _Stub(
        output=[
            _Stub(
                content=[
                    {
                        "type": "output_json",
                        "content": [{"type": "json_schema", "parsed": {"deep": 42}}],
                    }
                ]
            )
        ]
    )
    payload = runner._json_parser.extract_response_payload(response, "rtl")
    assert payload == {"deep": 42}


def test_extract_payload_from_messages_tree():
    runner = _make_runner()
    response = _Stub(
        messages=[
            _Stub(
                content=[
                    {
                        "type": "output_json",
                        "content": [{"type": "json_schema", "parsed": {"ok": True}}],
                    }
                ]
            )
        ]
    )
    payload = runner._json_parser.extract_response_payload(response, "rtl")
    assert payload == {"ok": True}


def test_deep_scan_does_not_return_list_for_object_stage_picks_dict():
    runner = _make_runner()
    # A list wrapper with two dicts; first is wrapper-like; second is payload-like.
    response = _Stub(
        output=[
            _Stub(
                content=[
                    [
                        {"type": "wrapper", "content": []},
                        {"name": "SpecX", "description": "...", "clock_mhz_target": 100,
                         "throughput_samples_per_cycle": 1,
                         "input_format": {"width": 16, "fractional_bits": 15},
                         "output_format": {"width": 16, "fractional_bits": 15},
                         "resource_budget": {"lut": 1, "ff": 1, "dsp": 1, "bram": 1},
                         "verification_config": {"num_samples": 1, "tolerance_abs": 0.1}}
                    ]
                ]
            )
        ]
    )
    payload = runner._json_parser.extract_response_payload(response, "spec")
    assert isinstance(payload, dict)
    assert payload.get("name") == "SpecX"


def test_extract_payload_invalid_text_raises():
    runner = _make_runner()
    response = _Stub(output=[_Stub(content=[_Stub(text="not json")])])
    with pytest.raises(RuntimeError):
        runner._json_parser.extract_response_payload(response, "rtl")
