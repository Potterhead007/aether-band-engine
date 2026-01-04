"""
AETHER Metrics Collection System

Production-grade metrics with:
- Counter, Gauge, Histogram, Timer
- Labels/tags support
- Aggregation and export
- Prometheus compatibility
"""

from __future__ import annotations

import functools
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from aether.core.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T", int, float)


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricLabel:
    """Label for metric dimensions."""

    name: str
    value: str

    def __hash__(self):
        return hash((self.name, self.value))


@dataclass
class MetricValue:
    """Timestamped metric value."""

    value: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """
    Counter metric that only increases.

    Usage:
        requests = Counter("http_requests_total", "Total HTTP requests")
        requests.inc()
        requests.inc(5, labels={"method": "GET", "status": "200"})
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the counter."""
        if amount < 0:
            raise ValueError("Counter can only be incremented")

        key = self._labels_to_key(labels)
        with self._lock:
            self._values[key] += amount

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> tuple:
        """Convert labels dict to hashable tuple."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        key = self._labels_to_key(labels)
        with self._lock:
            return self._values[key]

    def get_all(self) -> Dict[tuple, float]:
        """Get all counter values with labels."""
        with self._lock:
            return dict(self._values)

    def reset(self) -> None:
        """Reset counter (use with caution)."""
        with self._lock:
            self._values.clear()


class Gauge:
    """
    Gauge metric that can go up or down.

    Usage:
        temperature = Gauge("temperature_celsius", "Current temperature")
        temperature.set(23.5)
        temperature.inc(1.5)
        temperature.dec(0.5)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set the gauge to a value."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the gauge."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._values[key] += amount

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement the gauge."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._values[key] -= amount

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> tuple:
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        key = self._labels_to_key(labels)
        with self._lock:
            return self._values[key]

    def get_all(self) -> Dict[tuple, float]:
        """Get all gauge values with labels."""
        with self._lock:
            return dict(self._values)


class Histogram:
    """
    Histogram metric for distributions.

    Tracks count, sum, and buckets of observations.

    Usage:
        latency = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        latency.observe(0.234)
    """

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[tuple] = None,
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self.label_names = labels or []

        self._counts: Dict[tuple, int] = defaultdict(int)
        self._sums: Dict[tuple, float] = defaultdict(float)
        self._bucket_counts: Dict[tuple, Dict[float, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._observations: Dict[tuple, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record an observation."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._counts[key] += 1
            self._sums[key] += value
            self._observations[key].append(value)

            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[key][bucket] += 1

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> tuple:
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def get_count(self, labels: Optional[Dict[str, str]] = None) -> int:
        """Get observation count."""
        key = self._labels_to_key(labels)
        with self._lock:
            return self._counts[key]

    def get_sum(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get sum of observations."""
        key = self._labels_to_key(labels)
        with self._lock:
            return self._sums[key]

    def get_buckets(
        self, labels: Optional[Dict[str, str]] = None
    ) -> Dict[float, int]:
        """Get bucket counts."""
        key = self._labels_to_key(labels)
        with self._lock:
            return dict(self._bucket_counts[key])

    def get_percentile(
        self, percentile: float, labels: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """Calculate percentile from observations."""
        key = self._labels_to_key(labels)
        with self._lock:
            obs = self._observations[key]
            if not obs:
                return None
            sorted_obs = sorted(obs)
            idx = int(len(sorted_obs) * percentile / 100)
            return sorted_obs[min(idx, len(sorted_obs) - 1)]

    def get_stats(
        self, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Get statistical summary."""
        key = self._labels_to_key(labels)
        with self._lock:
            obs = self._observations[key]
            if not obs:
                return {
                    "count": 0,
                    "sum": 0,
                    "mean": 0,
                    "min": 0,
                    "max": 0,
                    "p50": 0,
                    "p90": 0,
                    "p95": 0,
                    "p99": 0,
                }

            sorted_obs = sorted(obs)
            return {
                "count": len(obs),
                "sum": sum(obs),
                "mean": statistics.mean(obs),
                "min": min(obs),
                "max": max(obs),
                "p50": self._percentile(sorted_obs, 50),
                "p90": self._percentile(sorted_obs, 90),
                "p95": self._percentile(sorted_obs, 95),
                "p99": self._percentile(sorted_obs, 99),
            }

    def _percentile(self, sorted_data: List[float], p: float) -> float:
        """Calculate percentile from sorted data."""
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]


class Timer:
    """
    Timer metric for measuring durations.

    Wrapper around Histogram optimized for timing.

    Usage:
        timer = Timer("operation_duration_seconds")

        # As context manager
        with timer.time():
            do_work()

        # As decorator
        @timer.time()
        def do_work():
            ...

        # Manual
        with timer.time(labels={"operation": "process"}):
            process()
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[tuple] = None,
        labels: Optional[List[str]] = None,
    ):
        self.histogram = Histogram(name, description, buckets, labels)
        self.name = name

    def time(
        self, labels: Optional[Dict[str, str]] = None
    ) -> "_TimerContext":
        """Create a timer context."""
        return _TimerContext(self.histogram, labels)

    def observe(self, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a duration directly."""
        self.histogram.observe(duration, labels)

    def get_stats(
        self, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Get timing statistics."""
        return self.histogram.get_stats(labels)


class _TimerContext:
    """Context manager for Timer."""

    def __init__(
        self,
        histogram: Histogram,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.histogram = histogram
        self.labels = labels
        self.start_time: Optional[float] = None

    def __enter__(self) -> "_TimerContext":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.histogram.observe(duration, self.labels)
        return False

    def __call__(self, func: F) -> F:
        """Use as decorator."""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with self:
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore


class MetricsCollector:
    """
    Central metrics collector and registry.

    Usage:
        metrics = MetricsCollector()

        # Register metrics
        requests = metrics.counter("http_requests_total")
        latency = metrics.histogram("http_request_duration_seconds")

        # Use metrics
        requests.inc()
        latency.observe(0.234)

        # Export
        all_metrics = metrics.collect()
    """

    # Global instance
    _instance: Optional["MetricsCollector"] = None

    def __init__(self, prefix: str = "aether"):
        self.prefix = prefix
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._timers: Dict[str, Timer] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls, prefix: str = "aether") -> "MetricsCollector":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(prefix)
        return cls._instance

    def _full_name(self, name: str) -> str:
        """Create full metric name with prefix."""
        if self.prefix:
            return f"{self.prefix}_{name}"
        return name

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """Get or create a counter metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._counters:
                self._counters[full_name] = Counter(full_name, description, labels)
            return self._counters[full_name]

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """Get or create a gauge metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._gauges:
                self._gauges[full_name] = Gauge(full_name, description, labels)
            return self._gauges[full_name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[tuple] = None,
        labels: Optional[List[str]] = None,
    ) -> Histogram:
        """Get or create a histogram metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._histograms:
                self._histograms[full_name] = Histogram(
                    full_name, description, buckets, labels
                )
            return self._histograms[full_name]

    def timer(
        self,
        name: str,
        description: str = "",
        buckets: Optional[tuple] = None,
        labels: Optional[List[str]] = None,
    ) -> Timer:
        """Get or create a timer metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._timers:
                self._timers[full_name] = Timer(
                    full_name, description, buckets, labels
                )
            return self._timers[full_name]

    def collect(self) -> Dict[str, Any]:
        """Collect all metrics."""
        result: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "counters": {},
            "gauges": {},
            "histograms": {},
            "timers": {},
        }

        with self._lock:
            for name, counter in self._counters.items():
                result["counters"][name] = {
                    "description": counter.description,
                    "values": {
                        str(k) if k else "default": v
                        for k, v in counter.get_all().items()
                    },
                }

            for name, gauge in self._gauges.items():
                result["gauges"][name] = {
                    "description": gauge.description,
                    "values": {
                        str(k) if k else "default": v
                        for k, v in gauge.get_all().items()
                    },
                }

            for name, histogram in self._histograms.items():
                result["histograms"][name] = {
                    "description": histogram.description,
                    "stats": histogram.get_stats(),
                    "buckets": histogram.get_buckets(),
                }

            for name, timer in self._timers.items():
                result["timers"][name] = {
                    "description": timer.histogram.description,
                    "stats": timer.get_stats(),
                }

        return result

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        with self._lock:
            # Counters
            for name, counter in self._counters.items():
                if counter.description:
                    lines.append(f"# HELP {name} {counter.description}")
                lines.append(f"# TYPE {name} counter")
                for labels, value in counter.get_all().items():
                    label_str = self._labels_to_prometheus(labels)
                    lines.append(f"{name}{label_str} {value}")

            # Gauges
            for name, gauge in self._gauges.items():
                if gauge.description:
                    lines.append(f"# HELP {name} {gauge.description}")
                lines.append(f"# TYPE {name} gauge")
                for labels, value in gauge.get_all().items():
                    label_str = self._labels_to_prometheus(labels)
                    lines.append(f"{name}{label_str} {value}")

            # Histograms
            for name, histogram in self._histograms.items():
                if histogram.description:
                    lines.append(f"# HELP {name} {histogram.description}")
                lines.append(f"# TYPE {name} histogram")
                stats = histogram.get_stats()
                lines.append(f"{name}_count {stats['count']}")
                lines.append(f"{name}_sum {stats['sum']}")
                for bucket, count in histogram.get_buckets().items():
                    lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')
                lines.append(f'{name}_bucket{{le="+Inf"}} {stats["count"]}')

        return "\n".join(lines)

    def _labels_to_prometheus(self, labels: tuple) -> str:
        """Convert labels tuple to Prometheus format."""
        if not labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in labels]
        return "{" + ",".join(pairs) + "}"

    def reset_all(self) -> None:
        """Reset all metrics (use with caution)."""
        with self._lock:
            for counter in self._counters.values():
                counter.reset()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()


# Convenience function
def get_metrics(prefix: str = "aether") -> MetricsCollector:
    """Get global metrics collector."""
    return MetricsCollector.get_instance(prefix)


# =============================================================================
# Pre-defined AETHER Metrics
# =============================================================================

def setup_default_metrics() -> MetricsCollector:
    """Set up default AETHER metrics."""
    metrics = get_metrics()

    # Pipeline metrics
    metrics.counter(
        "pipeline_runs_total",
        "Total pipeline runs",
        labels=["status", "genre"],
    )
    metrics.histogram(
        "pipeline_duration_seconds",
        "Pipeline execution duration",
        buckets=(1, 5, 10, 30, 60, 120, 300, 600),
        labels=["genre"],
    )
    metrics.gauge(
        "pipeline_active",
        "Number of active pipelines",
    )

    # Agent metrics
    metrics.counter(
        "agent_executions_total",
        "Total agent executions",
        labels=["agent_type", "status"],
    )
    metrics.histogram(
        "agent_duration_seconds",
        "Agent execution duration",
        labels=["agent_type"],
    )

    # Provider metrics
    metrics.counter(
        "provider_requests_total",
        "Total provider requests",
        labels=["provider", "operation", "status"],
    )
    metrics.histogram(
        "provider_latency_seconds",
        "Provider request latency",
        labels=["provider", "operation"],
    )
    metrics.gauge(
        "provider_circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=half-open, 2=open)",
        labels=["provider"],
    )

    # Audio metrics
    metrics.histogram(
        "audio_render_duration_seconds",
        "Audio rendering duration",
        labels=["format"],
    )
    metrics.counter(
        "audio_files_generated_total",
        "Total audio files generated",
        labels=["format"],
    )

    # QA metrics
    metrics.histogram(
        "qa_check_duration_seconds",
        "QA check duration",
        labels=["check_type"],
    )
    metrics.counter(
        "qa_checks_total",
        "Total QA checks",
        labels=["check_type", "result"],
    )

    return metrics
