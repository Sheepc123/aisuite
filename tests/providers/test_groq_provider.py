"""Tests for the Groq provider."""

from unittest.mock import MagicMock, patch
from copy import deepcopy
import json

import httpx
import pytest

from aisuite import Client
from aisuite.framework.message import Message
from aisuite.providers.groq_provider import GroqMessageConverter, GroqProvider


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("GROQ_API_KEY", "test-api-key")


def test_groq_provider():
    """Test that the provider is initialized and chat completions are requested."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = GroqProvider()
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [{"message": {"content": response_text_content}}]
    }

    with patch.object(
        provider.client.chat.completions,
        "create",
        return_value=mock_response,
    ) as mock_create:
        response = provider.chat_completions_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        mock_create.assert_called_with(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        assert response.choices[0].message.content == response_text_content


def test_groq_provider_with_usage():
    """Tests that usage data is correctly parsed when present in the response."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = GroqProvider()
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [
            {"message": {"content": response_text_content, "role": "assistant"}}
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }

    with patch.object(
        provider.client.chat.completions,
        "create",
        return_value=mock_response,
    ):
        response = provider.chat_completions_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30


@pytest.mark.parametrize("reasoning_content", [None, "internal reasoning"])
@pytest.mark.parametrize("as_dict", [False, True])
def test_groq_request_omits_reasoning_content_without_mutating_input(
    reasoning_content, as_dict
):
    message = Message(
        role="assistant", content="answer", reasoning_content=reasoning_content
    )
    if as_dict:
        message = message.model_dump(mode="json")
    original = deepcopy(message)

    converted = GroqMessageConverter.convert_request([message])

    assert "reasoning_content" not in converted[0]
    assert converted[0]["role"] == "assistant"
    assert converted[0]["content"] == "answer"
    assert message == original


def test_groq_automatic_tool_loop_replays_supported_assistant_fields():
    """Exercise the real SDK against a local transport, including the second turn."""
    requests = []
    tool_calls = [
        {
            "id": "call_weather",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
        }
    ]

    def handle_request(request):
        body = json.loads(request.content)
        requests.append(body)
        for message in body["messages"]:
            if message["role"] == "assistant" and "reasoning_content" in message:
                return httpx.Response(
                    400,
                    json={"error": {"message": "reasoning_content is unsupported"}},
                )
        message = (
            {"role": "assistant", "content": None, "tool_calls": tool_calls}
            if len(requests) == 1
            else {"role": "assistant", "content": "Sunny in Tokyo"}
        )
        return httpx.Response(
            200,
            json={
                "id": "chat_test",
                "object": "chat.completion",
                "created": 0,
                "model": "llama-3.3-70b-versatile",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if len(requests) == 1 else "stop",
                    }
                ],
            },
        )

    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as http_client:
        client = Client({"groq": {"http_client": http_client}})
        response = client.chat.completions.create(
            model="groq:llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Weather in Tokyo?"}],
            tools=[get_weather],
            max_turns=2,
        )

    assert len(requests) == 2
    assert requests[1]["messages"][1]["tool_calls"] == tool_calls
    assert requests[1]["messages"][2]["tool_call_id"] == "call_weather"
    assert json.loads(requests[1]["messages"][2]["content"]) == "Sunny in Tokyo"
    assert response.choices[0].message.content == "Sunny in Tokyo"
