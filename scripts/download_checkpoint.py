import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]


def main():
    registry = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
    models = registry["models"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[*models, "all"])
    args = parser.parse_args()

    selected = models if args.model == "all" else {args.model: models[args.model]}
    for files in selected.values():
        for filename in files.values():
            path = hf_hub_download(
                repo_id=registry["repo_id"],
                filename=filename,
                local_dir=str(ROOT),
            )
            print(path)


if __name__ == "__main__":
    main()
