import json
import sys

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

    ff_type = get_forcefield_type(config["forcefield"])
    print(f"Detected forcefield type: {ff_type}")

    # ---- Output dir ----
    output_dir = create_output_dir(config["output"]["directory"], ff_type)

    #---- Dataset ----
    td = config["training_data"]

    if td["source"] == "json":
        dataset = load_json_dataset(td)
        dataset = convert_to_descent(dataset, td)

    elif td["source"] == "qcarchive":
        dataset = load_qcarchive_dataset(td)
        dataset = convert_to_descent(dataset, td)

    elif td["source"] == "local":
        import datasets
        dataset = datasets.Dataset.load_from_disk(config["training_data"]["directory"])
        dataset.set_format('torch', columns=['energy', 'coords','forces'], output_all_columns=True)    

    # ---- Forcefield ----
    ff = setup_forcefield(config["forcefield"])
    ff = apply_functional_form(ff, ff_type)

    # ---- SMEE ----
    smee_ff, topologies = build_smee_system(dataset, ff)

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
