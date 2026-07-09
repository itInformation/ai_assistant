"""Tests for the Open-Meteo weather adapter."""

import asyncio
import json

import httpx
import pytest

from enterprise_ai_assistant.tools import OpenMeteoWeatherTool, ToolInputError


def json_response(request: httpx.Request, payload: object) -> httpx.Response:
    """Create an HTTPX response attached to its request."""

    return httpx.Response(200, request=request, content=json.dumps(payload).encode())


def test_weather_geocodes_and_maps_forecast() -> None:
    """The tool should combine geocoding and forecast responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "geocoding" in request.url.host:
            return json_response(
                request,
                {
                    "results": [
                        {
                            "name": "北京",
                            "admin1": "北京市",
                            "country": "中国",
                            "latitude": 39.9,
                            "longitude": 116.4,
                        }
                    ]
                },
            )
        return json_response(
            request,
            {
                "timezone": "Asia/Shanghai",
                "current": {
                    "time": "2026-07-09T12:00",
                    "temperature_2m": 30.5,
                    "apparent_temperature": 32.0,
                    "relative_humidity_2m": 50,
                    "precipitation": 0.0,
                    "weather_code": 1,
                    "wind_speed_10m": 8.0,
                },
                "daily": {
                    "time": ["2026-07-09"],
                    "weather_code": [1],
                    "temperature_2m_max": [31.0],
                    "temperature_2m_min": [22.0],
                    "precipitation_probability_max": [20],
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = OpenMeteoWeatherTool(client=client)

    result = asyncio.run(tool.invoke({"location": "北京", "forecast_days": 1}))
    asyncio.run(client.aclose())

    assert result.tool_name == "weather"
    assert result.data["current"]["temperature_c"] == 30.5
    assert result.data["daily"][0]["temperature_max_c"] == 31.0


def test_weather_rejects_unknown_location_and_extra_input() -> None:
    """Missing locations and undeclared fields should fail predictably."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(request, {})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = OpenMeteoWeatherTool(client=client)

    with pytest.raises(ToolInputError, match="not found"):
        asyncio.run(tool.invoke({"location": "不存在地点"}))
    with pytest.raises(ToolInputError, match="extra"):
        asyncio.run(tool.invoke({"location": "北京", "extra": True}))
    asyncio.run(client.aclose())
