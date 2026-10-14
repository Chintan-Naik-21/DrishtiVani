"""Supabase Storage Service.

Encapsulates all object storage interactions with Supabase Storage.
Ensures private buckets, uploads page renders and crops, generates time-limited signed URLs,
and performs connectivity checks. No other module should directly import the Supabase client.
"""

import logging
from typing import List, Optional
from supabase import Client, create_client
from app.config import get_settings

logger = logging.getLogger(__name__)


class StorageService:
    """Service wrapping Supabase Storage operations."""

    def __init__(self, client: Optional[Client] = None) -> None:
        """Initialize the Supabase client using credentials from configuration."""
        self._settings = get_settings()
        if client is not None:
            self._client = client
        elif self._settings.SUPABASE_URL and self._settings.SUPABASE_SERVICE_ROLE_KEY:
            self._client = create_client(
                self._settings.SUPABASE_URL,
                self._settings.SUPABASE_SERVICE_ROLE_KEY,
            )
            logger.info("Supabase Storage client initialized.")
        else:
            self._client = None
            logger.warning(
                "SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is not configured. "
                "Supabase Storage operations will fail until valid credentials are provided."
            )

    @property
    def client(self) -> Client:
        """Return the active Supabase client or raise if uninitialized."""
        if self._client is None:
            raise RuntimeError(
                "Supabase client is not configured. "
                "Please configure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in your .env file."
            )
        return self._client

    def ensure_buckets_exist(self) -> List[str]:
        """Verify that the required storage buckets exist; if not, create them as PRIVATE buckets.

        The three required buckets are:
        - handouts: original uploaded PDF handouts
        - pages: rendered page images (150 DPI)
        - crops: cropped equation and diagram images

        Returns:
            List[str]: List of ensured bucket names.
        """
        required_buckets = [
            self._settings.SUPABASE_BUCKET_HANDOUTS,
            self._settings.SUPABASE_BUCKET_PAGES,
            self._settings.SUPABASE_BUCKET_CROPS,
        ]

        try:
            existing = self.client.storage.list_buckets()
            existing_ids = {b.id for b in existing}
        except Exception as e:
            logger.error("Failed to list Supabase Storage buckets: %s", e)
            raise

        for bucket_name in required_buckets:
            if bucket_name not in existing_ids:
                logger.info("Bucket '%s' not found. Creating as PRIVATE bucket...", bucket_name)
                try:
                    # Supabase Storage creates private buckets by setting public=False
                    self.client.storage.create_bucket(
                        bucket_name,
                        options={"public": False},
                    )
                    logger.info("Successfully created private bucket '%s'.", bucket_name)
                except Exception as e:
                    # Check if another process created it concurrently
                    if "already exists" in str(e).lower():
                        logger.info("Bucket '%s' already exists.", bucket_name)
                    else:
                        logger.error("Failed to create bucket '%s': %s", bucket_name, e)
                        raise
            else:
                logger.debug("Bucket '%s' already exists.", bucket_name)

        return required_buckets

    def upload_file(
        self,
        bucket: str,
        path: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload raw bytes to a target bucket and storage path.

        Avoids storing duplicate file copies and overwrites if upsert is specified.

        Args:
            bucket: Name of the Supabase bucket.
            path: Relative storage path inside the bucket (e.g. 'doc-id/page_1.png').
            content: Raw file bytes.
            content_type: MIME type of the file.

        Returns:
            str: The storage path inside the bucket.
        """
        try:
            self.client.storage.from_(bucket).upload(
                path=path,
                file=content,
                file_options={"content-type": content_type, "x-upsert": "true", "upsert": "true"},
            )
            logger.debug("Uploaded %d bytes to '%s/%s'", len(content), bucket, path)
            return path
        except Exception as e:
            logger.error("Failed to upload file to '%s/%s': %s", bucket, path, e)
            raise

    def create_signed_url(
        self,
        bucket: str,
        path: Optional[str],
        expires_in: Optional[int] = None,
    ) -> Optional[str]:
        """Generate a time-limited signed URL for a file stored in a private bucket.

        Args:
            bucket: Name of the Supabase bucket.
            path: Storage path of the file in the bucket.
            expires_in: Expiration time in seconds (defaults to SIGNED_URL_EXPIRATION_SECONDS).

        Returns:
            Optional[str]: Time-limited signed URL, or None if path is None/empty.
        """
        if not path:
            return None

        ttl = expires_in if expires_in is not None else self._settings.SIGNED_URL_EXPIRATION_SECONDS
        try:
            res = self.client.storage.from_(bucket).create_signed_url(path, ttl)
            if isinstance(res, dict):
                return res.get("signedURL") or res.get("signedUrl")
            return str(res)
        except Exception as e:
            logger.error("Failed to create signed URL for '%s/%s': %s", bucket, path, e)
            return None

    def download_file(self, bucket: str, path: str) -> bytes:
        """Download raw bytes of a file from Supabase Storage.

        Args:
            bucket: Name of the Supabase bucket.
            path: Relative storage path.

        Returns:
            bytes: The downloaded file contents.
        """
        try:
            return self.client.storage.from_(bucket).download(path)
        except Exception as e:
            logger.error("Failed to download file '%s/%s': %s", bucket, path, e)
            raise

    def delete_file(self, bucket: str, path: str) -> None:
        """Delete a file from Supabase Storage.

        Args:
            bucket: Name of the Supabase bucket.
            path: Storage path to remove.
        """
        try:
            self.client.storage.from_(bucket).remove([path])
            logger.info("Deleted file '%s/%s'", bucket, path)
        except Exception as e:
            logger.error("Failed to delete file '%s/%s': %s", bucket, path, e)
            raise

    def check_storage_connectivity(self) -> bool:
        """Check if Supabase Storage is reachable by querying the bucket list.

        Returns:
            bool: True if storage is reachable, False otherwise.
        """
        try:
            self.client.storage.list_buckets()
            return True
        except Exception as e:
            logger.warning("Supabase Storage reachability check failed: %s", e)
            return False


_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Return a singleton instance of StorageService."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
