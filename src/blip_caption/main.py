from utils import *


def caption():
    """automated caption generation of a given dataset (all parameters are to set from config file)"""
    # Load BLIP processor and BLIP model
    processor, model = load_model()

    # Load absolute images paths
    paths = load_data()

    # Generate dataset captions
    captions = process(processor, model, paths)

    # Save captions with images paths in txt file
    save_captions(paths, captions)


if __name__ == "__main__":
    caption()
