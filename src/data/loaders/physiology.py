from pathlib import Path


class PhysiologyLoader:
    """
    Resolve physiology data files.

    Dataset-specific signal decoding will be added later.
    """

    def __init__(self, datasets_root: Path):
        self.datasets_root = datasets_root

    def load(self, relative_path: str):

        if not relative_path:
            raise ValueError(
                "Physiology path is empty."
            )

        path = self.datasets_root / relative_path

        if not path.exists():
            raise FileNotFoundError(
                f"Physiology file not found: {path}"
            )

        return {
            "path": str(path),
        }   