import argparse
import os
import re
import rai_wm


def main():
    parser = argparse.ArgumentParser(description="Realistic AI watermark util")

    parser.add_argument("--img", type=str, required=True, help="image embedding path (image minimal size: 512х768)")
    parser.add_argument("--wm", type=str, required=True, help="watermark string (hex, 32 length)")
    parser.add_argument("--pwd", type=str, required=True, help="password integer (1-65535)")
    parser.add_argument("--out", type=str, required=True, help="output image path")

    args = parser.parse_args()

    if not os.path.isfile(args.img):
        print(f"File doesn't exist: '{args.img}'")
        return
    
    if int(args.pwd) <= 0 or int(args.pwd) > 65535:
        print(f"Wrong `pwd` value: '{args.pwd}'")
        return
    
    if not re.match("^[0-9a-fA-F]{32}$", args.wm):
        print(f"Wrong `wm` value: '{args.wm}'")
        return

    rai_wm.embed(args.img, args.out, f"4b1{args.wm.lower()}32b", int(args.pwd))

if __name__ == "__main__":
    main()