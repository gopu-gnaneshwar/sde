from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.utils import safe_stem


@dataclass(slots=True)
class StoredFile:
    stored_filename: str
    path: Path


class FileStorage:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_upload(self, upload_file: UploadFile, *, prefix: str) -> StoredFile:
        original_name = upload_file.filename or "upload.bin"
        suffix = Path(original_name).suffix.lower()
        stem = safe_stem(Path(original_name).stem)
        stored_filename = f"{prefix}-{stem}{suffix}"
        destination = self.root / stored_filename

        upload_file.file.seek(0)
        with destination.open("wb") as handle:
            shutil.copyfileobj(upload_file.file, handle)

        return StoredFile(stored_filename=stored_filename, path=destination)
