from pathlib import Path

from xtrc.core.repo import walk_source_files


def test_repo_walk_ignores_build_dirs(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir(parents=True)
    (src / "main.py").write_text("print('hi')", encoding="utf-8")

    ignored = tmp_path / "node_modules"
    ignored.mkdir(parents=True)
    (ignored / "ignored.py").write_text("print('no')", encoding="utf-8")

    (tmp_path / ".gitignore").write_text("ignored_dir/\nignored_file.py\n", encoding="utf-8")
    ignored_dir = tmp_path / "ignored_dir"
    ignored_dir.mkdir(parents=True)
    (ignored_dir / "nested.py").write_text("print('skip')", encoding="utf-8")
    (tmp_path / "ignored_file.py").write_text("print('skip')", encoding="utf-8")

    files = walk_source_files(tmp_path)
    paths = {path.relative_to(tmp_path).as_posix() for path in files}
    assert "src/main.py" in paths
    assert "node_modules/ignored.py" not in paths
    assert "ignored_dir/nested.py" not in paths
    assert "ignored_file.py" not in paths
