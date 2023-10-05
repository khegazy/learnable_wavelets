import os
import argparse
import tests


parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    type=str,
    required=True,
    default="Name of test.py to run."
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.name not in tests.test_dict:
        raise ValueError(f"Cannot handle test name {args.name}, must be: {tests.test_dict.keys()}")
    tests.test_dict[args.name]()