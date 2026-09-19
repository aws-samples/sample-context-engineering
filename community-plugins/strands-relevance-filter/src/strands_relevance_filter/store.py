"""The offload store: the ``Store`` protocol and the in-memory, file, and S3 backends.

The plugin writes the raw sub-blocks of a filtered tool result to a ``Store`` before rewriting the
result, so every reference token embedded in the rewritten block resolves. A ``Store`` round-trips
content bytes and content type verbatim and raises ``KeyError`` for a reference that was never written.
"""

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError

__all__ = [
    "FileStore",
    "InMemoryStore",
    "S3Store",
    "Store",
]

logger = logging.getLogger(__name__)


def _sanitize_id(raw_id: str) -> str:
    """Sanitize an identifier for safe use in filenames and object keys.

    Path separators, parent-directory references, and any other unsafe character become underscores.

    Args:
        raw_id: The raw identifier.

    Returns:
        A string safe to embed in a filename or an S3 key.
    """
    sanitized = raw_id.replace("..", "_").replace("/", "_").replace("\\", "_")
    return re.sub(r"[^\w\-.]", "_", sanitized)


@runtime_checkable
class Store(Protocol):
    """Backend for the raw tool-result content this plugin offloads.

    Each sub-block of a filtered tool result is stored individually with its content type preserved.
    The plugin ships ``InMemoryStore`` (the default), ``FileStore``, and ``S3Store``; implement this
    protocol for any other backend (Redis, DynamoDB, ...).

    Lifecycle:
        The protocol carries no deletion or eviction method: stored content accumulates for the
        lifetime of the instance. For a long-running agent, use one instance per session or a backend
        with its own lifecycle management (an S3 lifecycle policy, ``InMemoryStore`` eviction).
    """

    async def store(self, key: str, content: bytes, content_type: str = "text/plain") -> str:
        """Store content and return a reference identifier.

        Args:
            key: A unique key for this content block.
            content: The raw content bytes to store.
            content_type: MIME type of the content (e.g. ``"text/plain"``, ``"application/json"``).

        Returns:
            A reference string that retrieves this content later.
        """
        ...

    async def retrieve(self, reference: str) -> tuple[bytes, str]:
        """Retrieve stored content by reference.

        Args:
            reference: The reference returned by a previous :meth:`store` call.

        Returns:
            A tuple of (content bytes, content type).

        Raises:
            KeyError: If the reference was never written (or is no longer held).
        """
        ...


class InMemoryStore:
    """Store offloaded content in memory. The default backend.

    Thread-safe, and useful anywhere disk access is unavailable or unwanted. References have the form
    ``mem_{counter}_{key}``.

    Supports cycle-based eviction: an entry neither stored nor retrieved within ``evict_after_turns``
    agent loop cycles is dropped. Eviction is on by default (20 cycles); pass ``None`` to disable it.

    Note:
        Content does not survive a process restart — use ``FileStore`` or ``S3Store`` for
        cross-session persistence. Each agent needs its own instance: sharing one across agents is
        rejected while eviction is enabled, because the cycle counter belongs to a single agent loop.

        An evicted entry is gone for good; the model retrieving it gets an error, and the original
        tool result is not kept in the conversation history after filtering — only the preview and its
        references remain in context.

    Args:
        evict_after_turns: Cycles of inactivity before an entry is evicted. Defaults to 20. ``None``
            disables eviction.
    """

    _DEFAULT_EVICT_AFTER_TURNS = 20

    def __init__(self, evict_after_turns: int | None = _DEFAULT_EVICT_AFTER_TURNS) -> None:
        """Initialize in-memory storage.

        Args:
            evict_after_turns: Cycles of inactivity before an entry is evicted. Defaults to 20.
                ``None`` disables eviction.

        Raises:
            ValueError: If ``evict_after_turns`` is not a positive integer.
        """
        if evict_after_turns is not None and evict_after_turns < 1:
            raise ValueError("evict_after_turns must be a positive integer")

        self._store: dict[str, tuple[bytes, str, int]] = {}
        self._counter: int = 0
        self._current_cycle: int = 0
        self._evict_after_turns: int | None = evict_after_turns
        self._bound_agent_id: int | None = None
        self._lock = threading.Lock()

    async def store(self, key: str, content: bytes, content_type: str = "text/plain") -> str:
        """Store content in memory and return a reference.

        Args:
            key: A unique key for this content block.
            content: The raw content bytes to store.
            content_type: MIME type of the content.

        Returns:
            A unique reference string of the form ``mem_{counter}_{key}``.
        """
        with self._lock:
            self._counter += 1
            reference = f"mem_{self._counter}_{key}"
            self._store[reference] = (content, content_type, self._current_cycle)
        return reference

    async def retrieve(self, reference: str) -> tuple[bytes, str]:
        """Retrieve content from memory.

        Refreshes the last-accessed cycle, so an entry the model keeps reading stays alive while
        eviction is enabled.

        Args:
            reference: The reference returned by :meth:`store`.

        Returns:
            A tuple of (content bytes, content type).

        Raises:
            KeyError: If the reference is not found, or was evicted.
        """
        with self._lock:
            if reference not in self._store:
                raise KeyError(f"Reference not found: {reference}")
            content, content_type, _ = self._store[reference]
            self._store[reference] = (content, content_type, self._current_cycle)
            return content, content_type

    def _bind(self, agent_id: int) -> None:
        """Claim this store for a single agent.

        Args:
            agent_id: Identity of the claiming agent.

        Raises:
            ValueError: If already bound to a different agent.
        """
        with self._lock:
            if self._bound_agent_id is None:
                self._bound_agent_id = agent_id
            elif self._bound_agent_id != agent_id:
                raise ValueError(
                    "InMemoryStore cannot be shared across multiple agents. "
                    "Use a separate InMemoryStore instance per agent."
                )

    def _evict(self, cycle: int) -> None:
        """Advance the current cycle and drop stale entries.

        Entries whose last-accessed cycle is more than ``evict_after_turns`` behind ``cycle`` are
        removed.

        Args:
            cycle: The agent's current event loop cycle count.
        """
        with self._lock:
            self._current_cycle = cycle
            if self._evict_after_turns is None:
                return
            threshold = cycle - self._evict_after_turns
            stale_refs = [ref for ref, (_, _, last_cycle) in self._store.items() if last_cycle < threshold]
            for ref in stale_refs:
                del self._store[ref]
            if stale_refs:
                logger.debug("evicted=<%d>, cycle=<%d> | stale entries removed", len(stale_refs), cycle)

    def clear(self) -> None:
        """Remove all stored content, freeing memory between sessions or after an invocation."""
        with self._lock:
            self._store.clear()


class FileStore:
    """Store offloaded content as files on the host filesystem.

    Files are written to the artifact directory under unique names, with the extension derived from
    the content type. A ``.metadata.json`` sidecar records the content type of every artifact, so
    types survive a process restart. The reference is the file path, preserving the form of
    ``artifact_dir``: a relative directory yields a relative reference, an absolute one an absolute
    reference.

    References are constrained to the artifact directory — a path containing ``..`` or pointing at a
    different parent directory is rejected with ``KeyError``.

    Args:
        artifact_dir: Directory where artifact files are written. Defaults to ``"./artifacts"``.
    """

    _METADATA_FILE = ".metadata.json"

    def __init__(self, artifact_dir: str = "./artifacts") -> None:
        """Initialize file-based storage.

        Args:
            artifact_dir: Directory where artifact files are written.
        """
        self._artifact_dir = Path(artifact_dir)
        self._counter: int = 0
        self._lock = threading.Lock()
        self._content_types: dict[str, str] = self._load_metadata()

    @staticmethod
    def _extension_for(content_type: str) -> str:
        """Return the file extension for a content type.

        Args:
            content_type: MIME type of the content.

        Returns:
            The extension, leading dot included.
        """
        if content_type == "text/plain":
            return ".txt"
        return f".{content_type.split('/')[-1]}"

    def _resolve_reference(self, reference: str) -> str:
        """Normalize a reference to a filename known to the metadata sidecar.

        Accepts full paths, bare filenames, and filename stems (no extension).

        Args:
            reference: Full path, bare filename, or filename stem.

        Returns:
            The resolved filename (basename with extension).

        Raises:
            KeyError: If the reference cannot be resolved, or escapes the artifact directory.
        """
        candidate = self._resolve_from_path(reference)

        if candidate in self._content_types:
            return candidate

        matches = [key for key in self._content_types if Path(key).stem == candidate]
        if len(matches) == 1:
            return matches[0]

        raise KeyError(f"Reference not found: {reference}")

    def _resolve_from_path(self, reference: str) -> str:
        """Validate a reference as a path inside the artifact directory.

        The fallback when :meth:`_resolve_reference` finds no metadata entry (sidecar missing or
        corrupt). Only full paths and bare filenames are accepted here — stem matching needs metadata.

        Args:
            reference: A full path or a bare filename.

        Returns:
            The validated filename (basename).

        Raises:
            KeyError: If the path lies outside the artifact directory.
        """
        if ".." in reference:
            raise KeyError(f"Reference not found: {reference}")
        ref_path = Path(reference)
        if len(ref_path.parts) > 1:
            if ref_path.parent.resolve() != self._artifact_dir.resolve():
                raise KeyError(f"Reference not found: {reference}")
            return ref_path.name
        return reference

    async def store(self, key: str, content: bytes, content_type: str = "text/plain") -> str:
        """Store content as a file and return its path as the reference.

        Args:
            key: A unique key for this content block.
            content: The raw content bytes to store.
            content_type: MIME type of the content.

        Returns:
            The file path (e.g. ``./artifacts/1234_1_key.txt``).

        Raises:
            OSError: If the directory or the file cannot be written.
        """
        sanitized_key = _sanitize_id(key)
        timestamp_ms = int(time.time() * 1000)
        ext = self._extension_for(content_type)

        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._counter += 1
            filename = f"{timestamp_ms}_{self._counter}_{sanitized_key}{ext}"
            self._content_types[filename] = content_type
            self._save_metadata()

        host_path = self._artifact_dir / filename
        host_path.write_bytes(content)
        return str(host_path)

    async def retrieve(self, reference: str) -> tuple[bytes, str]:
        """Retrieve content from a stored file.

        Accepts the full path returned by :meth:`store`, a bare filename, or a filename stem.

        Args:
            reference: The file path, filename, or stem returned by :meth:`store`.

        Returns:
            A tuple of (content bytes, content type). An artifact with no metadata entry reports
            ``"application/octet-stream"``.

        Raises:
            KeyError: If the reference does not resolve to a file inside the artifact directory.
        """
        try:
            filename = self._resolve_reference(reference)
        except KeyError:
            filename = self._resolve_from_path(reference)

        file_path = (self._artifact_dir / filename).resolve()
        if not file_path.is_relative_to(self._artifact_dir.resolve()) or not file_path.is_file():
            raise KeyError(f"Reference not found: {reference}")
        return file_path.read_bytes(), self._content_types.get(filename, "application/octet-stream")

    def _load_metadata(self) -> dict[str, str]:
        """Load the content-type sidecar, treating a missing or corrupt file as empty.

        Returns:
            The filename-to-content-type mapping.
        """
        metadata_path = self._artifact_dir / self._METADATA_FILE
        if metadata_path.is_file():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
            return loaded if isinstance(loaded, dict) else {}
        return {}

    def _save_metadata(self) -> None:
        """Write the content-type sidecar to the artifact directory."""
        metadata_path = self._artifact_dir / self._METADATA_FILE
        metadata_path.write_text(json.dumps(self._content_types), encoding="utf-8")


class S3Store:
    """Store offloaded content in Amazon S3.

    Objects are written under ``prefix`` in ``bucket`` with unique keys, and the content type is
    preserved as S3 object metadata. The reference is an ``s3://`` URI.

    Args:
        bucket: S3 bucket name.
        prefix: S3 key prefix organizing the stored artifacts.
        boto_session: Optional boto3 session. A new one is created from ``region_name`` when omitted.
        boto_client_config: Optional botocore client configuration. A caller-supplied
            ``user_agent_extra`` is extended with ``strands-agents`` rather than replaced.
        region_name: AWS region. Used only when ``boto_session`` is not given.

    Example:
        ```python
        from strands_relevance_filter import S3Store

        store = S3Store(bucket="my-agent-artifacts", prefix="tool-results/")
        ```
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        boto_session: boto3.Session | None = None,
        boto_client_config: BotocoreConfig | None = None,
        region_name: str | None = None,
    ) -> None:
        """Initialize S3-based storage.

        Args:
            bucket: S3 bucket name.
            prefix: S3 key prefix organizing the stored artifacts.
            boto_session: Optional boto3 session. A new one is created from ``region_name`` when
                omitted.
            boto_client_config: Optional botocore client configuration.
            region_name: AWS region. Used only when ``boto_session`` is not given.
        """
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        if self._prefix:
            self._prefix += "/"

        session = boto_session or boto3.Session(region_name=region_name)

        if boto_client_config:
            existing_user_agent = getattr(boto_client_config, "user_agent_extra", None)
            new_user_agent = f"{existing_user_agent} strands-agents" if existing_user_agent else "strands-agents"
            client_config = boto_client_config.merge(BotocoreConfig(user_agent_extra=new_user_agent))
        else:
            client_config = BotocoreConfig(user_agent_extra="strands-agents")

        self._client: Any = session.client(service_name="s3", config=client_config)
        self._counter: int = 0
        self._lock = threading.Lock()

    async def store(self, key: str, content: bytes, content_type: str = "text/plain") -> str:
        """Store content as an S3 object and return its ``s3://`` URI.

        Args:
            key: A unique key for this content block.
            content: The raw content bytes to store.
            content_type: MIME type of the content.

        Returns:
            An S3 URI (e.g. ``s3://bucket/prefix/1234_1_key``).

        Raises:
            botocore.exceptions.ClientError: If the S3 operation fails (missing bucket, permission
                denied, ...).
        """
        sanitized_key = _sanitize_id(key)
        timestamp_ms = int(time.time() * 1000)
        with self._lock:
            self._counter += 1
            counter = self._counter
        s3_key = f"{self._prefix}{timestamp_ms}_{counter}_{sanitized_key}"

        self._client.put_object(
            Bucket=self._bucket,
            Key=s3_key,
            Body=content,
            ContentType=content_type,
        )

        return f"s3://{self._bucket}/{s3_key}"

    async def retrieve(self, reference: str) -> tuple[bytes, str]:
        """Retrieve content from an S3 object.

        Accepts the ``s3://`` URI returned by :meth:`store` and a raw S3 key. References are
        constrained to the configured ``bucket`` and ``prefix``: one resolving to a different bucket
        or outside the prefix is rejected, mirroring the scope :meth:`store` writes within.

        Args:
            reference: The S3 URI or object key returned by :meth:`store`.

        Returns:
            A tuple of (content bytes, content type).

        Raises:
            KeyError: If the object does not exist, or the reference resolves outside the configured
                bucket and prefix.
            botocore.exceptions.ClientError: If the S3 operation fails for any other reason.
        """
        s3_key = reference
        if reference.startswith("s3://"):
            expected_prefix = f"s3://{self._bucket}/"
            if not reference.startswith(expected_prefix):
                raise KeyError(f"Reference not found: {reference}")
            s3_key = reference[len(expected_prefix) :]
        if self._prefix and not s3_key.startswith(self._prefix):
            raise KeyError(f"Reference not found: {reference}")
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=s3_key)
            content: bytes = response["Body"].read()
            content_type: str = response.get("ContentType", "application/octet-stream")
            return content, content_type
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise KeyError(f"Reference not found: {reference}") from e
            raise
