from pathlib import Path


class TextLoader:
    """
    Text loader.

    Tokenization belongs to the model/data-processing stage,
    not the metadata-resolution stage.
    """

    def __init__(self, datasets_root: Path | None = None):
        self.datasets_root = datasets_root

    def load(self, text=None, relative_path: str | None = None):
        """Load metadata text or explicitly file-backed transcript text."""
        if relative_path:
            if self.datasets_root is None:
                raise ValueError("A dataset root is required for file-backed text.")
            path = self.datasets_root / relative_path
            if not path.exists():
                raise FileNotFoundError(f"Text file not found: {path}")
            return path.read_text(encoding="utf-8", errors="replace")

        if text is None:
            return None

        return str(text)
