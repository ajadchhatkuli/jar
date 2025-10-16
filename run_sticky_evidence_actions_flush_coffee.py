"""Coffee-specific wrapper around the multiprocessing pipeline variant."""

import os

import run_sticky_evidence_actions_flush as pipeline
import run_sticky_evidence_actions_flush_multiproc as multiproc


pipeline.TITLE = "Coffee Preparation and Cleanup"
pipeline.VIDEO_PATH = "./realwear-videos/coffee-full.mp4"
pipeline.HOWTO_JSON = "./realwear-videos/howto_coffee.json"
pipeline.OBJECTS_JSON = "./realwear-videos/objects.json"
pipeline.ACTION_VERBS_JSON = "./realwear-videos/action_verbs.json"
pipeline.OUTPUT_DIR = "realwear-videos/outputs/coffee"
pipeline.S1_MAX_FRAMES = 5  # Limit Stage-1 prompts to five frames per window.


def main() -> None:
    os.makedirs(pipeline.OUTPUT_DIR, exist_ok=True)
    multiproc.main()


if __name__ == "__main__":
    main()
