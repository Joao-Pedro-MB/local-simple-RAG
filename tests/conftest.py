import pytest


@pytest.fixture
def notes(tmp_path):
    """Returns a function that writes a fixture Notes folder from {relative path: text}."""

    def write(files):
        root = tmp_path / "Notes"
        for relative_path, text in files.items():
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(text, bytes):
                path.write_bytes(text)
            else:
                path.write_text(text)
        root.mkdir(exist_ok=True)
        return root

    return write


@pytest.fixture
def index_dir(tmp_path):
    return tmp_path / "index"
