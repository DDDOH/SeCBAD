import argparse
from utils.helpers import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # TODO add args here

    return parser.parse_args(rest_args)
