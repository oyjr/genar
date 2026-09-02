import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parents[1]


def sample_ids(config):
    start, end = config["range"]
    excluded = set(config["exclude"])
    return [
        f'{config["prefix"]}{number}'
        for number in range(start, end + 1)
        if number not in excluded
    ]


def patterns(config):
    ids = sample_ids(config)
    return [f"st/{sample_id}.h5ad" for sample_id in ids] + [
        f"wsis/{sample_id}.*" for sample_id in ids
    ]


def prepare_metadata(config, output, gene_source):
    processed = output / config["directory"] / "processed_data"
    processed.mkdir(parents=True, exist_ok=True)
    slides = sample_ids(config)
    (processed / "all_slide_lst.txt").write_text(
        "".join(f"{slide_id}\n" for slide_id in slides),
        encoding="utf-8",
    )
    shutil.copyfile(gene_source, processed / "selected_gene_list.txt")
    return processed


def main():
    registry = json.loads(
        (ROOT / "configs" / "datasets.json").read_text(encoding="utf-8")
    )
    models = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
    datasets = registry["datasets"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=[*datasets, "all"])
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    selected = datasets if args.dataset == "all" else {
        args.dataset: datasets[args.dataset]
    }
    for name, config in selected.items():
        path = snapshot_download(
            repo_id=registry["repo_id"],
            repo_type="dataset",
            revision=registry["revision"],
            allow_patterns=patterns(config),
            local_dir=str(args.output / config["directory"]),
        )
        gene_source = hf_hub_download(
            repo_id=models["repo_id"],
            filename=models["models"][name]["genes"],
        )
        processed = prepare_metadata(config, args.output, Path(gene_source))
        print(path)
        print(processed)


if __name__ == "__main__":
    main()
