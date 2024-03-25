from animation import GenerativeMotion
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_path", type=str, default="put your pretrained path here")
parser.add_argument("--image", type=str, default=None)
parser.add_argument("--text", type=str, default=None)
parser.add_argument("--mask", type=str, default=None)
parser.add_argument("--eval", action="store_true")
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()
if args.eval:
    GenerativeMotion(args.pretrained_path, args.image, args.text, args.mask).render()