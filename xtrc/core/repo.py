from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from pathspec import PathSpec

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}

IGNORED_DIRS = {".git", "node_modules", "dist", "build", "__pycache__", ".xtrc"}
IGNORED_FILES = {".DS_Store"}


@dataclass(frozen=True)
class IgnoreMatcher:
    repo_path: Path
    spec: PathSpec | None

    @classmethod
    def from_repo(cls, repo_path: Path) -> "IgnoreMatcher":
        gitignore = repo_path / ".gitignore"
        if not gitignore.exists():
            return cls(repo_path=repo_path, spec=None)

        lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
        spec = PathSpec.from_lines("gitwildmatch", lines)
        return cls(repo_path=repo_path, spec=spec)

    def matches(self, candidate: Path, is_dir: bool) -> bool:
        if self.spec is None:
            return False
        rel = candidate.relative_to(self.repo_path).as_posix()
        if not rel or rel == ".":
            return False
        if self.spec.match_file(rel):
            return True
        if is_dir and self.spec.match_file(f"{rel}/"):
            return True
        return False


def detect_language(path: Path) -> str | None:
    return SUPPORTED_EXTENSIONS.get(path.suffix.lower())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def walk_source_files(repo_path: Path) -> list[Path]:
    ignore_matcher = IgnoreMatcher.from_repo(repo_path)
    files: list[Path] = []
    for root, dirs, filenames in os.walk(repo_path):
        root_path = Path(root)
        retained_dirs: list[str] = []
        for dirname in dirs:
            if dirname in IGNORED_DIRS or dirname.startswith("."):
                continue
            dir_path = root_path / dirname
            if ignore_matcher.matches(dir_path, is_dir=True):
                continue
            retained_dirs.append(dirname)
        dirs[:] = retained_dirs

        for name in filenames:
            if name in IGNORED_FILES or name.startswith("."):
                continue
            file_path = root_path / name
            if ignore_matcher.matches(file_path, is_dir=False):
                continue
            if detect_language(file_path) is not None:
                files.append(file_path)
    return files
