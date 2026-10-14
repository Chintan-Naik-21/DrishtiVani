"""Unit tests for StorageService."""

from unittest.mock import MagicMock
import pytest
from app.services.storage_service import StorageService


class FakeBucket:
    def __init__(self, bucket_id: str):
        self.id = bucket_id
        self.name = bucket_id


def test_ensure_buckets_exist_creates_missing_private_buckets():
    """Verify that ensure_buckets_exist detects missing buckets and creates them with public=False."""
    mock_client = MagicMock()
    mock_storage = MagicMock()
    mock_client.storage = mock_storage

    # Only 'handouts' exists initially; 'pages' and 'crops' are missing
    mock_storage.list_buckets.return_value = [FakeBucket("handouts")]

    service = StorageService(client=mock_client)
    ensured = service.ensure_buckets_exist()

    assert "handouts" in ensured
    assert "pages" in ensured
    assert "crops" in ensured

    # Verify create_bucket was called for 'pages' and 'crops' with options={"public": False}
    created_calls = mock_storage.create_bucket.call_args_list
    assert len(created_calls) == 2

    created_bucket_names = [call[0][0] for call in created_calls]
    assert "pages" in created_bucket_names
    assert "crops" in created_bucket_names

    for call in created_calls:
        assert call[1]["options"] == {"public": False}


def test_upload_file():
    """Verify upload_file passes file options and bytes to bucket proxy."""
    mock_client = MagicMock()
    mock_bucket_proxy = MagicMock()
    mock_client.storage.from_.return_value = mock_bucket_proxy

    service = StorageService(client=mock_client)
    res = service.upload_file("pages", "doc_1/p1.png", b"image-bytes", "image/png")

    assert res == "doc_1/p1.png"
    mock_client.storage.from_.assert_called_with("pages")
    mock_bucket_proxy.upload.assert_called_once()
    call_kwargs = mock_bucket_proxy.upload.call_args[1]
    assert call_kwargs["path"] == "doc_1/p1.png"
    assert call_kwargs["file"] == b"image-bytes"
    assert call_kwargs["file_options"]["content-type"] == "image/png"


def test_create_signed_url():
    """Verify signed URL creation returns expected URL string."""
    mock_client = MagicMock()
    mock_bucket_proxy = MagicMock()
    mock_bucket_proxy.create_signed_url.return_value = {
        "signedURL": "https://xyz.supabase.co/storage/v1/object/sign/pages/p1.png?token=abc"
    }
    mock_client.storage.from_.return_value = mock_bucket_proxy

    service = StorageService(client=mock_client)
    url = service.create_signed_url("pages", "doc_1/p1.png", expires_in=1800)

    assert url == "https://xyz.supabase.co/storage/v1/object/sign/pages/p1.png?token=abc"
    mock_bucket_proxy.create_signed_url.assert_called_once_with("doc_1/p1.png", 1800)


def test_create_signed_url_empty_path():
    """Verify create_signed_url returns None when path is empty or None."""
    mock_client = MagicMock()
    service = StorageService(client=mock_client)
    assert service.create_signed_url("crops", None) is None
    assert service.create_signed_url("crops", "") is None


def test_check_storage_connectivity_success():
    """Verify connectivity check returns True when bucket list succeeds."""
    mock_client = MagicMock()
    mock_client.storage.list_buckets.return_value = []
    service = StorageService(client=mock_client)
    assert service.check_storage_connectivity() is True


def test_check_storage_connectivity_failure():
    """Verify connectivity check returns False when an exception occurs."""
    mock_client = MagicMock()
    mock_client.storage.list_buckets.side_effect = ConnectionError("Supabase unreachable")
    service = StorageService(client=mock_client)
    assert service.check_storage_connectivity() is False
