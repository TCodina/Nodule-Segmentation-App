import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-workers",
    help="Number of worker processes for background data loading",
    default=8,
    type=int,
)

args = parser.parse_args()
print(args)
