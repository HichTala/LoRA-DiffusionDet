import argparse
import importlib

from dataset.coco_format import COCO


def main(args):
    dataset = importlib.import_module(f"configs.dataset.{args.dataset}")
    builder = COCO(args.dataset, dataset.ANNOTATIONS_PATH, dataset.IMAGES_PATH, dataset.FEATURES)
    builder.download_and_prepare()

    dataset = builder.as_dataset()
    dataset.push_to_hub(f"{args.hf_repo}/{args.dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Upload Dataset to Hub')
    parser.add_argument('hf_repo', type=str)
    parser.add_argument('dataset', type=str)

    main(parser.parse_args())
