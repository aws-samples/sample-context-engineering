"""Tests for the storage backends: InMemoryStore, FileStore, and S3Store."""

import json

import boto3
import pytest
from moto import mock_aws

from strands_relevance_filter.store import FileStore, InMemoryStore, S3Store

# --------------------------------------------------------------------------- #
# InMemoryStore                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_inmemory_round_trip_bytes_and_content_type():
    store = InMemoryStore()
    content = b"hello \x00\xff world"
    ref = await store.store("key1", content, content_type="text/plain")

    retrieved, content_type = await store.retrieve(ref)
    assert retrieved == content
    assert content_type == "text/plain"


@pytest.mark.asyncio
async def test_inmemory_round_trip_non_default_content_type():
    store = InMemoryStore()
    ref = await store.store("k", b'{"a": 1}', content_type="application/json")
    retrieved, content_type = await store.retrieve(ref)
    assert retrieved == b'{"a": 1}'
    assert content_type == "application/json"


@pytest.mark.asyncio
async def test_inmemory_unknown_reference_raises_keyerror():
    store = InMemoryStore()
    with pytest.raises(KeyError):
        await store.retrieve("mem_999_nope")


@pytest.mark.asyncio
async def test_inmemory_clear_empties_store():
    store = InMemoryStore()
    ref = await store.store("k", b"data")
    await store.retrieve(ref)  # confirm present
    store.clear()
    with pytest.raises(KeyError):
        await store.retrieve(ref)


@pytest.mark.asyncio
async def test_inmemory_same_key_twice_yields_distinct_references():
    """The docstring documents references of the form ``mem_{counter}_{key}``:
    the monotonic counter means the same key stored twice produces two distinct
    references, and both resolve independently rather than one overwriting the other.
    """
    store = InMemoryStore()
    ref1 = await store.store("dup", b"first", content_type="text/plain")
    ref2 = await store.store("dup", b"second", content_type="application/json")

    assert ref1 != ref2
    assert await store.retrieve(ref1) == (b"first", "text/plain")
    assert await store.retrieve(ref2) == (b"second", "application/json")


@pytest.mark.asyncio
async def test_inmemory_reject_invalid_evict_after_turns():
    with pytest.raises(ValueError):
        InMemoryStore(evict_after_turns=0)


# --------------------------------------------------------------------------- #
# FileStore                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_filestore_round_trip_text_plain(tmp_path):
    store = FileStore(artifact_dir=str(tmp_path))
    content = b"plain text body"
    ref = await store.store("doc", content, content_type="text/plain")

    retrieved, content_type = await store.retrieve(ref)
    assert retrieved == content
    assert content_type == "text/plain"
    assert ref.endswith(".txt")


@pytest.mark.asyncio
async def test_filestore_round_trip_application_json(tmp_path):
    store = FileStore(artifact_dir=str(tmp_path))
    content = b'{"key": "value"}'
    ref = await store.store("cfg", content, content_type="application/json")

    retrieved, content_type = await store.retrieve(ref)
    assert retrieved == content
    assert content_type == "application/json"
    assert ref.endswith(".json")


@pytest.mark.asyncio
async def test_filestore_round_trip_binary(tmp_path):
    store = FileStore(artifact_dir=str(tmp_path))
    content = b"\x89PNG\r\n\x1a\n\x00\x01\x02\xff\xfe"
    ref = await store.store("img", content, content_type="application/octet-stream")

    retrieved, content_type = await store.retrieve(ref)
    assert retrieved == content
    assert content_type == "application/octet-stream"


@pytest.mark.asyncio
async def test_filestore_file_appears_on_disk(tmp_path):
    store = FileStore(artifact_dir=str(tmp_path))
    ref = await store.store("k", b"on disk", content_type="text/plain")

    from pathlib import Path

    file_path = Path(ref)
    assert file_path.is_file()
    assert file_path.read_bytes() == b"on disk"
    assert file_path.parent.resolve() == tmp_path.resolve()


@pytest.mark.asyncio
async def test_filestore_unknown_reference_raises_keyerror(tmp_path):
    store = FileStore(artifact_dir=str(tmp_path))
    with pytest.raises(KeyError):
        await store.retrieve(str(tmp_path / "never_written.txt"))


@pytest.mark.asyncio
async def test_filestore_second_instance_resolves_via_metadata(tmp_path):
    """The .metadata.json sidecar lets a fresh FileStore over the same directory
    resolve a reference (and its content type) written by an earlier instance.
    """
    first = FileStore(artifact_dir=str(tmp_path))
    ref = await first.store("shared", b'{"n": 2}', content_type="application/json")

    # A metadata sidecar must exist.
    metadata_path = tmp_path / ".metadata.json"
    assert metadata_path.is_file()
    assert isinstance(json.loads(metadata_path.read_text()), dict)

    second = FileStore(artifact_dir=str(tmp_path))
    retrieved, content_type = await second.retrieve(ref)
    assert retrieved == b'{"n": 2}'
    assert content_type == "application/json"


@pytest.mark.asyncio
async def test_filestore_resolves_by_bare_filename_and_stem(tmp_path):
    store = FileStore(artifact_dir=str(tmp_path))
    ref = await store.store("k", b"body", content_type="text/plain")

    from pathlib import Path

    filename = Path(ref).name
    stem = Path(ref).stem

    assert (await store.retrieve(filename))[0] == b"body"
    assert (await store.retrieve(stem))[0] == b"body"


@pytest.mark.asyncio
async def test_filestore_rejects_path_traversal(tmp_path):
    store = FileStore(artifact_dir=str(tmp_path))
    with pytest.raises(KeyError):
        await store.retrieve("../escape.txt")


# --------------------------------------------------------------------------- #
# S3Store                                                                     #
# --------------------------------------------------------------------------- #


# NOTE: mock_aws is used as a context manager INSIDE each async body rather than
# as an outer decorator: layered above @pytest.mark.asyncio it wraps the coroutine
# function in a plain sync function, so pytest-asyncio no longer sees a coroutine
# and never awaits the test.


@pytest.mark.asyncio
async def test_s3store_round_trip():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        store = S3Store(bucket="test-bucket", prefix="artifacts/", region_name="us-east-1")

        content = b"s3 payload \x00\xff"
        ref = await store.store("obj", content, content_type="application/json")
        assert ref.startswith("s3://test-bucket/artifacts/")

        retrieved, content_type = await store.retrieve(ref)
        assert retrieved == content
        assert content_type == "application/json"


@pytest.mark.asyncio
async def test_s3store_retrieve_by_raw_key():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        store = S3Store(bucket="test-bucket", prefix="artifacts/", region_name="us-east-1")

        ref = await store.store("obj", b"body", content_type="text/plain")
        raw_key = ref[len("s3://test-bucket/") :]

        retrieved, content_type = await store.retrieve(raw_key)
        assert retrieved == b"body"
        assert content_type == "text/plain"


@pytest.mark.asyncio
async def test_s3store_unknown_reference_raises_keyerror():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        store = S3Store(bucket="test-bucket", prefix="artifacts/", region_name="us-east-1")

        with pytest.raises(KeyError):
            await store.retrieve("s3://test-bucket/artifacts/does-not-exist")


@pytest.mark.asyncio
async def test_s3store_reference_outside_bucket_raises_keyerror():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        store = S3Store(bucket="test-bucket", prefix="artifacts/", region_name="us-east-1")

        with pytest.raises(KeyError):
            await store.retrieve("s3://other-bucket/artifacts/x")


@pytest.mark.asyncio
async def test_s3store_reference_outside_prefix_raises_keyerror():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        store = S3Store(bucket="test-bucket", prefix="artifacts/", region_name="us-east-1")

        with pytest.raises(KeyError):
            await store.retrieve("s3://test-bucket/elsewhere/x")
