"""G2b L1 preparation and leader-approved freeze. No work on import."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_freeze import main

if __name__ == '__main__':
    print(main())
