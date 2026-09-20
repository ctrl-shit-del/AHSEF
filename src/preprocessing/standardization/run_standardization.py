from pathlib import Path

import pandas as pd

from src.preprocessing.standardization.standardizer import (
    standardize,
)
from src.preprocessing.standardization.validator import (
    validate,
)


def main():

    df = standardize()

    validate(df)

    print("\nStandardization completed successfully.")


if __name__ == "__main__":
    main()