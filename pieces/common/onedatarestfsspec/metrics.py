"""OpenTelemetry metrics integration for OnedataRESTFSSpec."""

import logging
import os
import uuid
from typing import Any, Dict, NamedTuple, Optional

logger = logging.getLogger(__name__)


class _Instruments(NamedTuple):
    """Container for all OpenTelemetry meter instruments."""

    access_total: Any
    read_bytes: Any
    written_bytes: Any
    read_duration: Any
    write_duration: Any
    throughput: Any


try:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    _OTEL_SDK_AVAILABLE = True
except ImportError:
    _OTEL_SDK_AVAILABLE = False


def _build_exporter(endpoint: Optional[str], protocol: str) -> Any:
    """Instantiate the appropriate OTLP metric exporter for *protocol*.

    Parameters
    ----------
    endpoint : str, optional
        Override the OTLP collector endpoint.  When *None* the exporter reads
        ``OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`` / ``OTEL_EXPORTER_OTLP_ENDPOINT``
        from the environment automatically.
    protocol : str
        ``"grpc"`` or any ``"http/*"`` variant (default ``"http/protobuf"``).
    """
    kwargs: Dict[str, Any] = {}
    if endpoint:
        kwargs["endpoint"] = endpoint

    if protocol == "grpc":
        try:
            # pylint: disable=import-outside-toplevel
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter as GrpcOTLPMetricExporter,
            )
        except ImportError as exc:
            raise ImportError(
                "opentelemetry-exporter-otlp-proto-grpc is required for the gRPC "
                "protocol.  Install it with: "
                "pip install opentelemetry-exporter-otlp-proto-grpc"
            ) from exc
        return GrpcOTLPMetricExporter(**kwargs)

    # http/protobuf (default) or http/json
    try:
        # pylint: disable=import-outside-toplevel
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter as HttpOTLPMetricExporter,
        )
    except ImportError as exc:
        raise ImportError(
            "opentelemetry-exporter-otlp-proto-http is required for the HTTP "
            "protocol.  Install it with: "
            "pip install opentelemetry-exporter-otlp-proto-http"
        ) from exc
    return HttpOTLPMetricExporter(**kwargs)


class OnedataMetrics:
    """OpenTelemetry metrics collector for OnedataFileSystem operations.

    Instruments are created once per instance and report the following metrics,
    each labeled with ``space_id``, ``file_id``, ``provider_id``,
    ``session_id``, and ``operation`` (``"read"`` or ``"write"``):

    * ``onedata_file_access_total``            – counter, total read + write ops
    * ``onedata_read_bytes``                   – counter, cumulative bytes read
    * ``onedata_written_bytes``                – counter, cumulative bytes written
    * ``onedata_read_duration``                – histogram, read latency (seconds)
    * ``onedata_write_duration``               – histogram, write latency (seconds)
    * ``onedata_file_throughput_bytes_per_second`` – histogram, observed throughput

    Parameters
    ----------
    enabled : bool
        Activate metric collection.  When *False* every ``record_*`` call is
        a no-op.  Can also be enabled via the ``ONEDATA_METRICS_ENABLED=true``
        environment variable (kwargs take precedence).
    endpoint : str, optional
        Override OTLP collector endpoint.  When omitted the exporter reads
        ``OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`` / ``OTEL_EXPORTER_OTLP_ENDPOINT``
        from the environment.
    protocol : str, optional
        ``"grpc"`` or ``"http/protobuf"`` (default).  Falls back to
        ``OTEL_EXPORTER_OTLP_PROTOCOL`` then ``"http/protobuf"``.
    export_interval_ms : int
        How often the periodic reader flushes metrics to the collector
        (milliseconds, default 60 000).
    session_id : str, optional
        Identifier attached to every metric as the ``session_id`` attribute.
        When *None* a random UUID is generated automatically and reused for
        the lifetime of this ``OnedataMetrics`` instance.  Can also be set
        via the ``ONEDATA_OTLP_SESSION_ID`` environment variable (the
        constructor argument takes precedence).
    """

    def __init__(
        self,
        enabled: bool = False,
        endpoint: Optional[str] = None,
        protocol: Optional[str] = None,
        export_interval_ms: int = 60_000,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        self.enabled = False
        self.session_id: str = (
            session_id
            or os.environ.get("ONEDATA_OTLP_SESSION_ID", "")
            or str(uuid.uuid4())
        )

        if not enabled:
            return

        if not _OTEL_SDK_AVAILABLE:
            logger.warning(
                "opentelemetry-sdk is not installed; metrics are disabled.  "
                "Install with: pip install 'onedatarestfsspec[monitoring]'"
            )
            return

        resolved_protocol = (
            protocol or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
        ).strip()

        try:
            exporter = _build_exporter(endpoint, resolved_protocol)
            reader = PeriodicExportingMetricReader(
                exporter, export_interval_millis=export_interval_ms
            )
            self._provider = MeterProvider(metric_readers=[reader])
            meter = self._provider.get_meter(
                "onedatarestfsspec",
                schema_url="https://opentelemetry.io/schemas/1.24.0",
            )
            self._instruments = _Instruments(
                access_total=meter.create_counter(
                    "onedata_file_access_total",
                    unit="1",
                    description=(
                        "Total number of file access operations. "
                        "The 'operation' attribute is either 'read' or 'write'."
                    ),
                ),
                read_bytes=meter.create_counter(
                    "onedata_read_bytes",
                    unit="By",
                    description="Total bytes read from Onedata",
                ),
                written_bytes=meter.create_counter(
                    "onedata_written_bytes",
                    unit="By",
                    description="Total bytes written to Onedata",
                ),
                read_duration=meter.create_histogram(
                    "onedata_read_duration",
                    unit="s",
                    description="Latency of read operations in seconds",
                ),
                write_duration=meter.create_histogram(
                    "onedata_write_duration",
                    unit="s",
                    description="Latency of write operations in seconds",
                ),
                throughput=meter.create_histogram(
                    "onedata_file_throughput_bytes_per_second",
                    unit="By/s",
                    description="Observed data transfer throughput in bytes per second",
                ),
            )
            self.enabled = True
            logger.debug(
                "OpenTelemetry metrics enabled (protocol=%s)", resolved_protocol
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to initialise OpenTelemetry metrics: %s", exc)

    def record_read(
        self,
        space_id: str,
        file_id: str,
        provider_id: str,
        *,
        byte_count: int,
        latency_s: float,
    ) -> None:
        """Record a completed read operation.

        Parameters
        ----------
        space_id : str
            Onedata space identifier (file ID of the space root directory).
        file_id : str
            Onedata file identifier.
        provider_id : str
            Domain of the Oneprovider that served the request.
        byte_count : int
            Number of bytes transferred.
        latency_s : float
            Wall-clock duration of the operation in seconds.
        """
        if not self.enabled:
            return
        attrs = {
            "space_id": space_id,
            "file_id": file_id,
            "provider_id": provider_id,
            "session_id": self.session_id,
            "operation": "read",
        }
        self._instruments.access_total.add(1, attrs)
        self._instruments.read_bytes.add(byte_count, attrs)
        self._instruments.read_duration.record(latency_s, attrs)
        if latency_s > 0:
            self._instruments.throughput.record(byte_count / latency_s, attrs)

    def record_write(
        self,
        space_id: str,
        file_id: str,
        provider_id: str,
        *,
        byte_count: int,
        latency_s: float,
    ) -> None:
        """Record a completed write operation.

        Parameters
        ----------
        space_id : str
            Onedata space identifier (file ID of the space root directory).
        file_id : str
            Onedata file identifier.
        provider_id : str
            Domain of the Oneprovider that served the request.
        byte_count : int
            Number of bytes transferred.
        latency_s : float
            Wall-clock duration of the operation in seconds.
        """
        if not self.enabled:
            return
        attrs = {
            "space_id": space_id,
            "file_id": file_id,
            "provider_id": provider_id,
            "session_id": self.session_id,
            "operation": "write",
        }
        self._instruments.access_total.add(1, attrs)
        self._instruments.written_bytes.add(byte_count, attrs)
        self._instruments.write_duration.record(latency_s, attrs)
        if latency_s > 0:
            self._instruments.throughput.record(byte_count / latency_s, attrs)

    def shutdown(self) -> None:
        """Flush pending metrics and shut down the provider."""
        if self.enabled and hasattr(self, "_provider"):
            self._provider.shutdown()
