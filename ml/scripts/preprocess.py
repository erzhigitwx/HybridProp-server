import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)

from ml.config import load_config
from ml.data.preprocessor import DataPreprocessor
from ml.utils.device import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Preprocess Yelp data")
    parser.add_argument("--config", type=str, default="ml/config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    preprocessor = DataPreprocessor(cfg)
    tables = preprocessor.run()

    print("\n=== Preprocessing Summary ===")
    for name, df in tables.items():
        print(f"  {name:20s}: {len(df):>8,} rows")


if __name__ == "__main__":
    main()
