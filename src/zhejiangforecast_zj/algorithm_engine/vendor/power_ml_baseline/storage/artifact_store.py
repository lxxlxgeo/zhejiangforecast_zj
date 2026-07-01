from __future__ import annotations

from pathlib import Path
import logging
import shutil

LOGGER = logging.getLogger(__name__)


class ArtifactStore:
    def __init__(self, cfg: dict | None):
        self.cfg = cfg or {}
        self.kind = str(self.cfg.get("type", "local")).lower()

    def upload_dir(self, local_dir: str | Path, remote_prefix: str | None = None) -> None:
        local_dir = Path(local_dir)
        if self.kind in {"none", "local"}:
            dest_root = self.cfg.get("mirror_dir")
            if dest_root:
                dest = Path(dest_root) / (remote_prefix or local_dir.name)
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(local_dir, dest)
            return
        if self.kind in {"s3", "minio"}:
            self._upload_s3_dir(local_dir, remote_prefix or local_dir.name)
            return
        raise ValueError(f"Unsupported artifact store type: {self.kind}")

    def _upload_s3_dir(self, local_dir: Path, remote_prefix: str) -> None:
        try:
            import boto3
        except Exception as exc:
            raise RuntimeError("boto3 is required for S3/MinIO artifact upload") from exc
        bucket = self.cfg["bucket"]
        endpoint_url = self.cfg.get("endpoint_url")
        kwargs = {}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        if self.cfg.get("aws_access_key_id"):
            kwargs["aws_access_key_id"] = self.cfg.get("aws_access_key_id")
        if self.cfg.get("aws_secret_access_key"):
            kwargs["aws_secret_access_key"] = self.cfg.get("aws_secret_access_key")
        s3 = boto3.client("s3", **kwargs)
        for path in local_dir.rglob("*"):
            if path.is_file():
                key = f"{remote_prefix}/{path.relative_to(local_dir).as_posix()}"
                LOGGER.info("Uploading %s to s3://%s/%s", path, bucket, key)
                s3.upload_file(str(path), bucket, key)
