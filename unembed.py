import argparse
import os
import re
import rai_wm


def main():
    parser = argparse.ArgumentParser(description="Realistic AI watermark util")

    parser.add_argument("--img", type=str, required=True, help="watermarked image path")
    parser.add_argument("--pwd", type=str, required=True, help="password integer (1-65535)")

    args = parser.parse_args()

    if not os.path.isfile(args.img):
        print(f"File doesn't exist: '{args.img}'")
        return

    print(rai_wm.unembed(args.img, int(args.pwd), ("4b1", "32b")))

if __name__ == "__main__":
    main()