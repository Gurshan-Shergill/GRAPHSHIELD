import io
import os
from typing import BinaryIO, Optional
from minio import Minio
from minio.error import S3Error

from app.core.config import get_settings

settings = get_settings()


class StorageClient:
    def __init__(self):
        if settings.s3_endpoint and settings.s3_access_key and settings.s3_secret_key:
            self.client = Minio(
                settings.s3_endpoint.replace('https://', '').replace('http://', ''),
                access_key=settings.s3_access_key,
                secret_key=settings.s3_secret_key,
                secure=settings.s3_endpoint.startswith('https'),
                region=settings.s3_region,
            )
            self.bucket = settings.s3_bucket or settings.minio_bucket
        else:
            self.client = Minio(
                settings.minio_endpoint,
                access_key=settings.minio_root_user,
                secret_key=settings.minio_root_password,
                secure=settings.minio_secure,
            )
            self.bucket = settings.minio_bucket

    def ensure_bucket(self) -> None:
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except S3Error:
            pass

    def upload_file(
        self,
        object_name: str,
        data: BinaryIO,
        length: int,
        content_type: str = 'application/octet-stream',
    ) -> str:
        self.ensure_bucket()
        self.client.put_object(
            self.bucket,
            object_name,
            data,
            length,
            content_type=content_type,
        )
        return f's3://{self.bucket}/{object_name}'

    def upload_bytes(
        self,
        object_name: str,
        data: bytes,
        content_type: str = 'application/octet-stream',
    ) -> str:
        self.ensure_bucket()
        self.client.put_object(
            self.bucket,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type=content_type,
        )
        return f's3://{self.bucket}/{object_name}'

    def download_file(self, object_name: str) -> bytes:
        response = self.client.get_object(self.bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def download_stream(self, object_name: str):
        return self.client.get_object(self.bucket, object_name)

    def delete_file(self, object_name: str) -> bool:
        try:
            self.client.remove_object(self.bucket, object_name)
            return True
        except S3Error:
            return False

    def file_exists(self, object_name: str) -> bool:
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error:
            return False

    def get_presigned_url(self, object_name: str, expires: int = 3600) -> str:
        return self.client.presigned_get_object(self.bucket, object_name, expires=expires)


storage = StorageClient()
