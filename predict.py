import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()    


if __name__=="__main__":
    main()
