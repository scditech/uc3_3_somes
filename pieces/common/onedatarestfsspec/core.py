"""Core implementation of OnedataFileSystem for fsspec."""

import logging
import os
import posixpath
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from fsspec import AbstractFileSystem  # type: ignore[import-untyped]
from fsspec.registry import register_implementation  # type: ignore[import-untyped]
from fsspec.spec import AbstractBufferedFile  # type: ignore[import-untyped]
from fsspec.utils import infer_storage_options  # type: ignore[import-untyped]

from onedatafilerestclient import OnedataFileRESTClient
from onedatafilerestclient.errors import OnedataError
from onedatafilerestclient.types import FileId

from .config import get_onedata_config_from_env, merge_config
from .metrics import OnedataMetrics
from .utils import split_onedata_path

logger = logging.getLogger(__name__)


class OnedataFile(AbstractBufferedFile):  # type: ignore[misc]
    """File-like object for Onedata files."""

    def __init__(
        self,
        fs: "OnedataFileSystem",
        path: str,
        mode: str = "rb",
        *,
        block_size: Optional[int] = None,
        cache_type: str = "readahead",
        **kwargs: Any,
    ) -> None:
        super().__init__(fs, path, mode, block_size, cache_type=cache_type, **kwargs)
        self.space_name, self.file_path = self._split_onedata_path(path)
        self.file_id = None

        if self.mode == "rb":
            try:
                self.size = fs._get_file_size(self.space_name, self.file_path)
                self.file_id = fs._get_file_id(  # pylint: disable=protected-access
                    self.space_name, self.file_path
                )
            except OnedataError:
                if "r" in self.mode:
                    raise
                self.size = 0

    def _split_onedata_path(self, path: str) -> Tuple[str, str]:
        """Split Onedata path into space name and file path."""
        space_name, file_path = split_onedata_path(path)
        return space_name, file_path or ""

    def _fetch_range(self, start: int, end: int) -> bytes:
        """Fetch a range of bytes from the file."""
        if self.file_id is None:
            # pylint: disable=protected-access
            self.file_id = self.fs._get_file_id(self.space_name, self.file_path)

        size = end - start
        t0 = time.monotonic()
        content = self.fs.client.get_file_content(
            self.space_name, file_id=self.file_id, offset=start, size=size
        )
        result = bytes(content) if content is not None else b""
        # pylint: disable=protected-access
        if self.fs.metrics.enabled:
            space_id = self.fs._get_space_id(self.space_name)
            provider_id = self.fs._get_provider_id(self.space_name)
        else:
            space_id = provider_id = ""
        self.fs.metrics.record_read(
            space_id,
            str(self.file_id),
            provider_id,
            byte_count=len(result),
            latency_s=time.monotonic() - t0,
        )
        return result

    def _upload_chunk(self, final: bool = False) -> bool:
        """Upload buffered data to Onedata."""
        if self.buffer is None or not self.buffer.tell():
            return False

        if self.file_id is None:
            # Create the file if it doesn't exist
            self.file_id = self.fs._create_file(  # pylint: disable=protected-access
                self.space_name, self.file_path
            )

        data = self.buffer.getvalue()
        t0 = time.monotonic()
        self.fs.client.put_file_content(
            self.space_name, data=data, file_id=self.file_id, offset=self.offset
        )
        # pylint: disable=protected-access
        if self.fs.metrics.enabled:
            space_id = self.fs._get_space_id(self.space_name)
            provider_id = self.fs._get_provider_id(self.space_name)
        else:
            space_id = provider_id = ""
        self.fs.metrics.record_write(
            space_id,
            str(self.file_id),
            provider_id,
            byte_count=len(data),
            latency_s=time.monotonic() - t0,
        )

        if self.fs.auto_mkdir and self.path.count("/") > 1:
            self.fs.makedirs(posixpath.dirname(self.path), exist_ok=True)

        self.offset += len(data)
        self.buffer.seek(0)
        self.buffer.truncate()

        return True

    def commit(self) -> None:
        """Commit any remaining data."""
        if self.mode not in {"rb", "ab"}:
            self._upload_chunk(final=True)
        self.discard()

    def discard(self) -> None:
        """Discard the file."""


class OnedataFileSystem(AbstractFileSystem):  # type: ignore[misc]  # pylint: disable=too-many-instance-attributes
    """fsspec filesystem implementation for Onedata."""

    protocol = "onedata"
    root_marker = "/"

    def __init__(
        self,
        *,
        onezone_host: Optional[str] = None,
        token: Optional[str] = None,
        preferred_providers: Optional[List[str]] = None,
        verify_ssl: bool = True,
        timeout: Optional[Union[float, Tuple[float, float]]] = 30,
        auto_mkdir: bool = True,
        metrics_enabled: bool = False,
        otlp_endpoint: Optional[str] = None,
        otlp_protocol: Optional[str] = None,
        otlp_export_interval_ms: int = 60_000,
        otlp_session_id: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize OnedataFileSystem.

        Parameters
        ----------
        onezone_host : str
            Onedata Onezone host URL
        token : str
            Onedata access token
        preferred_providers : list of str, optional
            List of preferred Oneprovider domains
        verify_ssl : bool, default True
            Whether to verify SSL certificates
        timeout : float or tuple, default 30
            Connection timeout
        auto_mkdir : bool, default True
            Whether to automatically create parent directories
        metrics_enabled : bool, default False
            Enable OpenTelemetry metrics export.  Can also be activated by
            setting the ``ONEDATA_METRICS_ENABLED=true`` environment variable.
            Requires ``opentelemetry-sdk`` and an OTLP exporter to be installed
            (``pip install 'onedatarestfsspec[monitoring]'``).
        otlp_endpoint : str, optional
            Override the OTLP collector endpoint.  When omitted the exporter
            reads ``OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`` /
            ``OTEL_EXPORTER_OTLP_ENDPOINT`` from the environment.
        otlp_protocol : str, optional
            Transport protocol: ``"grpc"`` or ``"http/protobuf"`` (default).
            Falls back to the ``OTEL_EXPORTER_OTLP_PROTOCOL`` environment
            variable, then ``"http/protobuf"``.
        otlp_export_interval_ms : int, default 60000
            How often the periodic metric reader flushes to the collector
            (milliseconds).
        otlp_session_id : str, optional
            Session identifier attached to every metric as the ``session_id``
            attribute.  When omitted the value is read from the
            ``ONEDATA_OTLP_SESSION_ID`` environment variable; if that is also
            unset a random UUID is generated once and reused for the lifetime
            of this filesystem instance.
        """
        super().__init__(**kwargs)

        config = self._resolve_connection_config(
            onezone_host, token, preferred_providers, verify_ssl, timeout
        )

        self.onezone_host = config.get("onezone_host")
        self.token = config.get("token")
        self.preferred_providers = config.get("preferred_providers", [])
        self.verify_ssl = config.get("verify_ssl", True)
        self.timeout = config.get("timeout", 30)
        self.auto_mkdir = auto_mkdir

        if not self.onezone_host or not self.token:
            raise ValueError(
                "Both onezone_host and token must be provided either as "
                "parameters or environment variables"
            )

        self.client = OnedataFileRESTClient(
            onezone_host=self._extract_hostname(self.onezone_host),
            token=self.token,
            preferred_providers=self.preferred_providers,
            verify_ssl=self.verify_ssl,
            timeout=self.timeout,
        )

        # Metrics — honor both the kwarg and the env-var override
        self.metrics = OnedataMetrics(
            enabled=metrics_enabled
            or (os.environ.get("ONEDATA_METRICS_ENABLED", "").lower() == "true"),
            endpoint=otlp_endpoint,
            protocol=otlp_protocol,
            export_interval_ms=otlp_export_interval_ms,
            session_id=otlp_session_id,
        )

        # Cache for space name → space ID (file ID of the space root directory)
        self._space_id_cache: Dict[str, str] = {}

    @staticmethod
    def _resolve_connection_config(
        onezone_host: Optional[str],
        token: Optional[str],
        preferred_providers: Optional[List[str]],
        verify_ssl: bool,
        timeout: Any,
    ) -> Dict[str, Any]:
        """Merge env-var defaults with explicitly provided constructor kwargs."""
        env_config = get_onedata_config_from_env()
        explicit_config = {
            "onezone_host": onezone_host,
            "token": token,
            "preferred_providers": preferred_providers,
            "verify_ssl": verify_ssl,
            "timeout": timeout,
        }
        return merge_config({}, env_config, explicit_config)

    @staticmethod
    def _extract_hostname(host: str) -> str:
        """Return the bare hostname/IP from *host*, stripping any URL scheme."""
        if host.startswith(("http://", "https://")):
            parsed = urlparse(host)
            return parsed.hostname or parsed.netloc
        return host

    @classmethod
    def _strip_protocol(cls, path: str) -> str:
        """Remove the protocol from a path."""
        if path.startswith("onedata://"):
            options = infer_storage_options(path)
            return str(options["path"])
        return path

    def _split_onedata_path(self, path: str) -> Tuple[str, Optional[str]]:
        """Split a path into space name and file path."""
        return split_onedata_path(path)

    def _get_file_id(self, space_name: str, file_path: Optional[str] = None) -> FileId:
        """Get file ID for a given space and path."""
        file_id = self.client.get_file_id(space_name, file_path=file_path)
        return str(file_id)

    def _get_space_id(self, space_name: str) -> str:
        """Return the Onedata space ID for *space_name*, resolving it on first use.

        The space ID is the file ID of the space root directory.  Results are
        cached per filesystem instance to avoid repeated round-trips.
        """
        if space_name not in self._space_id_cache:
            self._space_id_cache[space_name] = self.client.get_space_id(space_name)
        return self._space_id_cache[space_name]

    def _get_provider_id(self, space_name: str) -> str:
        """Return the id of the Oneprovider selected for *space_name*.

        Delegates to the provider selector so that the result reflects the
        same priority ordering (preferred providers, version, graylisting)
        used by the REST client when dispatching actual requests.  The
        underlying token-scope lookup is cached by the Onezone client for
        a short period, so repeated calls within that window are cheap.
        """
        try:
            # pylint: disable=protected-access
            providers = self.client._provider_selector.list_available_space_providers(
                space_name, oz_rest_client=self.client._oz_client
            )
            return providers[0].id if providers else ""
        except Exception:  # pylint: disable=broad-except
            return ""

    def _resolve_metric_labels(
        self, space_name: str, file_path: Optional[str]
    ) -> Tuple[str, str, str]:
        """Return ``(space_id, file_id, provider_id)`` for metric attributes.

        Each value falls back to an empty string when the corresponding lookup
        fails, so that a resolution error never breaks the actual I/O operation.
        """
        space_id = ""
        file_id = ""
        try:
            space_id = self._get_space_id(space_name)
        except Exception:  # pylint: disable=broad-except
            pass
        try:
            file_id = self._get_file_id(space_name, file_path)
        except Exception:  # pylint: disable=broad-except
            pass
        return space_id, file_id, self._get_provider_id(space_name)

    def _get_file_size(self, space_name: str, file_path: Optional[str] = None) -> int:
        """Get file size for a given space and path."""
        attrs = self.client.get_attributes(
            space_name, file_path=file_path, attributes=["size"]
        )
        size_value = attrs["size"]
        return int(size_value) if size_value is not None else 0

    def _create_file(self, space_name: str, file_path: str) -> FileId:
        """Create a new file and return its ID."""
        file_id = self.client.create_file(
            space_name,
            file_path=file_path,
            file_type="REG",
            create_parents=self.auto_mkdir,
        )
        return str(file_id)

    def ls(
        self, path: str, detail: bool = False, **kwargs: Any
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """List contents of a directory.

        Parameters
        ----------
        path : str
            Path to list
        detail : bool, default False
            Whether to return detailed information

        Returns
        -------
        list
            List of file names or detailed file information
        """
        path = self._strip_protocol(path).rstrip("/")
        space_name, file_path = self._split_onedata_path(path)

        if not space_name:
            # List all spaces
            spaces = self.client.list_spaces()
            if detail:
                return [
                    {"name": space, "type": "directory", "size": 0} for space in spaces
                ]
            return spaces

        try:
            result = self.client.list_children(
                space_name,
                file_path=file_path,
                attributes=["name", "type", "size", "mtime", "posixPermissions"],
            )

            files = []
            for child in result["children"]:
                name = child["name"]
                full_path = (
                    f"{space_name}/{file_path}/{name}"
                    if file_path
                    else f"{space_name}/{name}"
                )

                if detail:
                    files.append(
                        {
                            "name": full_path,
                            "type": "directory" if child["type"] == "DIR" else "file",
                            "size": child.get("size", 0),
                            "mtime": child.get("mtime"),
                            "mode": child.get("posixPermissions"),
                        }
                    )
                else:
                    files.append(full_path)  # type: ignore[arg-type]

            return files

        except OnedataError as e:
            if "enoent" in str(e).lower():
                raise FileNotFoundError(f"Path not found: {path}") from e
            raise

    def info(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        """Get file/directory information.

        Parameters
        ----------
        path : str
            Path to get info for

        Returns
        -------
        dict
            File information
        """
        path = self._strip_protocol(path).rstrip("/")
        space_name, file_path = self._split_onedata_path(path)

        if not space_name:
            raise FileNotFoundError("Root path info not available")

        try:
            attrs = self.client.get_attributes(
                space_name,
                file_path=file_path,
                attributes=[
                    "name",
                    "type",
                    "size",
                    "mtime",
                    "atime",
                    "posixPermissions",
                ],
            )

            return {
                "name": path,
                "type": "directory" if attrs["type"] == "DIR" else "file",
                "size": attrs.get("size", 0),
                "mtime": attrs.get("mtime"),
                "atime": attrs.get("atime"),
                "mode": attrs.get("posixPermissions"),
            }

        except OnedataError as e:
            if "enoent" in str(e).lower():
                raise FileNotFoundError(f"Path not found: {path}") from e
            raise

    def cat_file(
        self,
        path: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        **kwargs: Any,
    ) -> bytes:
        """Read file content.

        Parameters
        ----------
        path : str
            Path to read
        start : int, optional
            Start byte position
        end : int, optional
            End byte position

        Returns
        -------
        bytes
            File content
        """
        path = self._strip_protocol(path)
        space_name, file_path = self._split_onedata_path(path)

        if not space_name or not file_path:
            raise FileNotFoundError(f"Invalid path: {path}")

        metric_labels = (
            self._resolve_metric_labels(space_name, file_path)
            if self.metrics.enabled
            else ("", "", "")
        )

        try:
            t0 = time.monotonic()
            if start is not None or end is not None:
                # Get file size for bounds checking
                file_size = self._get_file_size(space_name, file_path)
                start = start or 0
                end = end or file_size
                size = end - start
                content = self.client.get_file_content(
                    space_name, file_path=file_path, offset=start, size=size
                )
            else:
                content = self.client.get_file_content(space_name, file_path=file_path)

            result = bytes(content) if content is not None else b""
            self.metrics.record_read(
                *metric_labels,
                byte_count=len(result),
                latency_s=time.monotonic() - t0,
            )
            return result

        except OnedataError as e:
            if "enoent" in str(e).lower():
                raise FileNotFoundError(f"File not found: {path}") from e
            raise

    def cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        """Copy a file within Onedata."""
        data = self.cat_file(path1)

        path2 = self._strip_protocol(path2)
        space_name, file_path = self._split_onedata_path(path2)

        if not space_name or not file_path:
            raise ValueError(f"Invalid destination path: {path2}")

        file_id = self._create_file(space_name, file_path)
        self.client.put_file_content(space_name, data=data, file_id=file_id)

    def rm_file(self, path: str) -> None:
        """Remove a file.

        Parameters
        ----------
        path : str
            Path to remove
        """
        path = self._strip_protocol(path)
        space_name, file_path = self._split_onedata_path(path)

        if not space_name or not file_path:
            raise ValueError(f"Invalid path: {path}")

        try:
            self.client.remove(space_name, file_path=file_path)
        except OnedataError as e:
            if "enoent" in str(e).lower():
                raise FileNotFoundError(f"File not found: {path}") from e
            raise

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        """Create directories.

        Parameters
        ----------
        path : str
            Directory path to create
        exist_ok : bool, default False
            Don't raise error if directory exists
        """
        path = self._strip_protocol(path).rstrip("/")
        space_name, dir_path = self._split_onedata_path(path)

        if not space_name or not dir_path:
            return  # Can't create spaces

        try:
            self.client.create_file(
                space_name, file_path=dir_path, file_type="DIR", create_parents=True
            )
        except OnedataError as e:
            if not exist_ok or "eexist" not in str(e).lower():
                raise

    def rmdir(self, path: str) -> None:
        """Remove a directory.

        Parameters
        ----------
        path : str
            Directory path to remove
        """
        self.rm_file(path)

    def exists(self, path: str, **kwargs: Any) -> bool:
        """Check if a path exists.

        Parameters
        ----------
        path : str
            Path to check

        Returns
        -------
        bool
            True if path exists
        """
        try:
            self.info(path)
            return True
        except FileNotFoundError:
            return False

    def isdir(self, path: str) -> bool:
        """Check if a path is a directory.

        Parameters
        ----------
        path : str
            Path to check

        Returns
        -------
        bool
            True if path is a directory
        """
        try:
            info = self.info(path)
            return bool(info["type"] == "directory")
        except FileNotFoundError:
            return False

    def isfile(self, path: str) -> bool:
        """Check if a path is a file.

        Parameters
        ----------
        path : str
            Path to check

        Returns
        -------
        bool
            True if path is a file
        """
        try:
            info = self.info(path)
            return bool(info["type"] == "file")
        except FileNotFoundError:
            return False

    def size(self, path: str) -> int:
        """Get file size.

        Parameters
        ----------
        path : str
            File path

        Returns
        -------
        int
            File size in bytes
        """
        info = self.info(path)
        size_value = info["size"]
        return int(size_value) if size_value is not None else 0

    def open(  # pylint: disable=arguments-differ
        self,
        path: str,
        mode: str = "rb",
        *,
        block_size: Optional[int] = None,  # pylint: disable=unused-argument
        cache_type: Optional[str] = None,  # pylint: disable=unused-argument
        compression: Optional[str] = None,  # pylint: disable=unused-argument
        **kwargs: Any,
    ) -> OnedataFile:
        """Open a file for reading or writing.

        Parameters
        ----------
        path : str
            File path
        mode : str, default "rb"
            File open mode

        Returns
        -------
        OnedataFile
            File handle
        """
        return OnedataFile(self, path, mode, **kwargs)

    def _rm(self, path: str) -> None:
        """Remove a file or directory."""
        self.rm_file(path)

    def created(self, path: str) -> Optional[float]:
        """Get file creation time (not supported)."""
        return None

    def modified(self, path: str) -> Optional[float]:
        """Get file modification time."""
        try:
            info = self.info(path)
            return info.get("mtime")
        except FileNotFoundError:
            return None

    @property
    def fsid(self) -> str:
        """Get filesystem ID."""
        return "onedata"

    @property
    def otlp_session_id(self) -> str:
        """Return the OTLP session ID used to label all metrics for this instance."""
        return self.metrics.session_id

    def sign(self, path: str, expiration: int = 3600, **kwargs: Any) -> str:
        """Sign a path (not implemented)."""
        raise NotImplementedError("URL signing not supported")


# Register with fsspec
try:
    register_implementation("onedata", OnedataFileSystem)
except ImportError:
    pass  # fsspec not available
