import json
import sys
import datasets
from datetime import datetime
import pathlib
import descent.targets.energy


from fit import (
    get_forcefield_type,
    load_qcarchive_dataset,
    load_json_dataset,
    convert_to_descent,
    setup_forcefield,
    apply_functional_form,
    build_smee_system,
    create_trainable,
    train_model,
    update_forcefield_parameters,
    create_output_dir,
    save_forcefield,
)


def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ff_type = get_forcefield_type(config["forcefield"])
    print(f"Detected forcefield type: {ff_type}")

    # ---- Output dir ----
    output_dir = create_output_dir(config["output"]["directory"], ff_type)

    #---- Dataset ----
    td = config["training_data"]

    if td["source"] == "local":
        if td["type"] == "QC":
            dataset = datasets.Dataset.load_from_disk(td["directory"])
        elif td["type"] == "mlip":
            import torch

            path = pathlib.Path(td["directory"])

            metadata_file = path / "metadata.json"
            entries_file = path / "entries.pt"

            if not entries_file.exists():
                raise FileNotFoundError(f"Missing entries.pt in {path}")

            print(f"Loading MLIP dataset from {path}")

            entries = torch.load(entries_file)

            print(f"Loaded {len(entries)} entries")

            dataset = descent.targets.energy.create_dataset(entries=entries)

            dataset.set_format(
                "torch",
                columns=["energy", "coords", "forces"],
                output_all_columns=True
            )
    else:
        all_datasets = []
        all_stats = []

        if td["source"] == "json":
            for sub_config in td["datasets"]:
                collection = load_json_dataset(sub_config)
                ds, stats = convert_to_descent(collection, sub_config, output_dir)
                all_datasets.append(ds)
                all_stats.append(stats)

        elif td["source"] == "qcarchive":
            for sub_config in td["datasets"]:
                collection = load_qcarchive_dataset(sub_config)
                ds, stats = convert_to_descent(collection, sub_config, output_dir)
                all_datasets.append(ds)
                all_stats.append(stats)
        
        dataset = datasets.concatenate_datasets(all_datasets)

        combined_stats = {
            "torsiondrive_total": sum(s["torsiondrive"]["total_records"] for s in all_stats),
            "torsiondrive_saved": sum(s["torsiondrive"]["saved_records"] for s in all_stats),
            "optimization_total": sum(s["optimization"]["total_records"] for s in all_stats),
            "optimization_saved": sum(s["optimization"]["saved_records"] for s in all_stats),
        }

        combined_stats["total_input_records"] = (
            combined_stats["torsiondrive_total"]
            + combined_stats["optimization_total"]
        )

        combined_stats["total_saved_records"] = (
            combined_stats["torsiondrive_saved"]
            + combined_stats["optimization_saved"]
        )

        stats_path = output_dir / "dataset_stats.json"  
        with open(stats_path, "w") as f:
            json.dump(combined_stats, f, indent=2)

        print(f"Saved dataset stats to: {stats_path}")
        print(json.dumps(combined_stats, indent=2))

    dataset.set_format(
        'torch',
        columns=['energy', 'coords', 'forces'],
        output_all_columns=True
    )

    save_path = pathlib.Path("datasets") / f"combined_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(save_path))
    print(f"Saved combined dataset to: {save_path}")

    # ---- Forcefield ----
    ff = setup_forcefield(config["forcefield"])
    ff = apply_functional_form(ff, ff_type)

    # ---- SMEE ----
    smee_ff, topologies, valid_smiles = build_smee_system(dataset, ff)
    dataset = dataset.filter(lambda x: x["smiles"] in set(valid_smiles))
    
    # ---- Trainable ----
    trainable = create_trainable(smee_ff, ff_type)

    # ---- Training ----
    final_ff, trained_params = train_model(
        trainable,
        dataset,
        topologies,
        config["training"],
        output_dir
    )

    # Write back parameters
    updated_ff = update_forcefield_parameters(
        smee_force_field=smee_ff,
        trainable=trainable,
        base_forcefield=ff
    )

    # ---- Save ----
    save_forcefield(updated_ff, output_dir, ff_type)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py input.json")
        sys.exit(1)

    main(sys.argv[1])
