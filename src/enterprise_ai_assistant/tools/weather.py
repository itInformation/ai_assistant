"""Open-Meteo weather tool with geocoding and bounded forecasts."""

from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential

from enterprise_ai_assistant.core import get_logger
from enterprise_ai_assistant.models import JSONValue, ToolResult, ToolSpec
from enterprise_ai_assistant.tools._validation import validate_arguments
from enterprise_ai_assistant.tools.exceptions import (
    ToolInputError,
    ToolProviderError,
)

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherInput(BaseModel):
    """Validated arguments accepted by the weather tool."""

    model_config = ConfigDict(extra="forbid")

    location: str = Field(min_length=2, max_length=100)
    forecast_days: int = Field(default=3, ge=1, le=7)


class _RetryableWeatherError(Exception):
    """Internal marker for retryable provider responses."""


class OpenMeteoWeatherTool:
    """Resolve a place name and return current and daily weather."""

    spec = ToolSpec(
        name="weather",
        description="查询指定城市当前天气及未来 1 至 7 天预报。",
        parameters=WeatherInput.model_json_schema(),
    )

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("timeout must be positive and retries non-negative")
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()
        self._logger = get_logger(tool="weather", provider="open-meteo")

    async def invoke(
        self,
        arguments: Mapping[str, JSONValue],
    ) -> ToolResult:
        """Look up coordinates and fetch a compact weather forecast."""

        request = validate_arguments(WeatherInput, arguments)
        try:
            location = await self._get_json(
                _GEOCODING_URL,
                {
                    "name": request.location,
                    "count": 1,
                    "language": "zh",
                    "format": "json",
                },
            )
            results = location.get("results")
            if not isinstance(results, list) or not results:
                raise ToolInputError("weather location was not found")
            place = results[0]
            forecast = await self._get_json(
                _FORECAST_URL,
                {
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "timezone": "auto",
                    "forecast_days": request.forecast_days,
                    "current": (
                        "temperature_2m,relative_humidity_2m,"
                        "apparent_temperature,precipitation,"
                        "weather_code,wind_speed_10m"
                    ),
                    "daily": (
                        "weather_code,temperature_2m_max,"
                        "temperature_2m_min,precipitation_probability_max"
                    ),
                },
            )
            data = self._map_response(place, forecast)
        except ToolInputError:
            raise
        except (
            httpx.HTTPError,
            _RetryableWeatherError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            self._logger.error("weather_request_failed", error_type=type(exc).__name__)
            raise ToolProviderError(
                f"weather provider failed: {type(exc).__name__}"
            ) from exc
        return ToolResult(
            tool_name=self.spec.name,
            content=(
                f"{data['location']}天气: 当前 " f"{data['current']['temperature_c']}°C"
            ),
            data=data,
            metadata={"provider": "Open-Meteo"},
        )

    async def close(self) -> None:
        """Close the internally owned HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    async def _get_json(
        self,
        url: str,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.TransportError, _RetryableWeatherError)
            ),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                response = await self._client.get(
                    url,
                    params=params,
                    timeout=self._timeout_seconds,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise _RetryableWeatherError(
                        f"retryable HTTP status {response.status_code}"
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("weather response must be an object")
                return payload
        raise AssertionError("retry loop completed without a result")

    @staticmethod
    def _map_response(
        place: Mapping[str, Any],
        forecast: Mapping[str, Any],
    ) -> dict[str, JSONValue]:
        current = forecast["current"]
        daily = forecast["daily"]
        dates = daily["time"]
        days = [
            {
                "date": dates[index],
                "weather_code": daily["weather_code"][index],
                "temperature_max_c": daily["temperature_2m_max"][index],
                "temperature_min_c": daily["temperature_2m_min"][index],
                "precipitation_probability_max": daily["precipitation_probability_max"][
                    index
                ],
            }
            for index in range(len(dates))
        ]
        return {
            "location": ", ".join(
                str(value)
                for value in (
                    place.get("name"),
                    place.get("admin1"),
                    place.get("country"),
                )
                if value
            ),
            "latitude": float(place["latitude"]),
            "longitude": float(place["longitude"]),
            "timezone": str(forecast["timezone"]),
            "current": {
                "time": str(current["time"]),
                "temperature_c": float(current["temperature_2m"]),
                "apparent_temperature_c": float(current["apparent_temperature"]),
                "relative_humidity_percent": int(current["relative_humidity_2m"]),
                "precipitation_mm": float(current["precipitation"]),
                "weather_code": int(current["weather_code"]),
                "wind_speed_kmh": float(current["wind_speed_10m"]),
            },
            "daily": days,
        }
