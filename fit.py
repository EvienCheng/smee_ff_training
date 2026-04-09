import os
import math
import json
import pathlib
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import tqdm
import more_itertools
import tensorboardX
import datasets

from qcportal import PortalClient
from openff.units import unit
from openff.qcsubmit.results import (
    TorsionDriveResultCollection,
    OptimizationResultCollection,
)

from openff.toolkit import ForceField, Molecule, Topology
import openff.interchange as interchange

import smee.converters
import descent.targets.energy
import descent.train


def get_forcefield_type(ff):
    ff = ForceField(
        ff,
        load_plugins=True,
        allow_cosmetic_attributes=True
    )

    # print(ff.registered_parameter_handlers)

    if "LeeKrimm" in ff.registered_parameter_handlers:
        return "LeeKrimm"

    elif "HarmonicAngle" in ff.registered_parameter_handlers:
        return "HarmonicAngle"

    elif "HarmonicHeight" in ff.registered_parameter_handlers:
        return "HarmonicHeight"

    elif "ImproperTorsions" in ff.registered_parameter_handlers:
        return "TwoMinima"

    return "Unknown"

# Dataset Loading
import descent.targets.energy
from qcportal import PortalClient
from openff.qcsubmit.results import (
    TorsionDriveResultCollection,
    OptimizationResultCollection,
)
from openff.units import unit
import numpy as np

def load_json_dataset(config):
    data_type = config["type"]
    directory = config["directory"]

    if data_type == "torsiondrive":
        collection = TorsionDriveResultCollection.parse_file(
            directory
        )

    elif data_type == "optimization":
        collection = OptimizationResultCollection.parse_file(
            directory        
        )

    return collection

def load_qcarchive_dataset(config):
    client = PortalClient("https://api.qcarchive.molssi.org:443")

    data_type = config["type"]
    dataset_name = config["dataset"]
    spec_name = config.get("spec_name", "default")

    print(f"Loading {data_type} dataset: {dataset_name}")

    if data_type == "torsiondrive":
        collection = TorsionDriveResultCollection.from_server(
            client=client,
            datasets=dataset_name,
            spec_name=spec_name,
        )

    elif data_type == "optimization":
        collection = OptimizationResultCollection.from_server(
            client=client,
            datasets=dataset_name,
            spec_name=spec_name,
        )

    else:
        raise ValueError(f"Unknown dataset type: {data_type}")

    return collection

def convert_to_descent(collection, config):
    data_type = config["type"]

    bohr_to_angstrom = (1 * unit.bohr).m_as(unit.angstrom)
    hartree_to_kcal = (1 * unit.hartree * unit.avogadro_constant).m_as(
        unit.kilocalories_per_mole
    )

    records_and_molecules = list(collection.to_records())

    print(f"Total records: {len(records_and_molecules)}")

    descent_entries = []
    data_by_smiles = defaultdict(list)

    # print(records_and_molecules)

    count = 0
    for record, molecule in tqdm.tqdm(records_and_molecules):

        # TorsionDrive
        if data_type == "torsiondrive":

            for opt in record.minimum_optimizations.values():
                last = opt.trajectory[-1]
                last_mol = last.molecule

                mapped_smiles = last_mol.identifiers.canonical_isomeric_explicit_hydrogen_mapped_smiles
                if mapped_smiles is None:
                    continue
                
                coords = last_mol.geometry * bohr_to_angstrom
                energy = last.properties["return_energy"] * hartree_to_kcal

                gradient = np.array(
                    last.properties["scf total gradient"]
                ).reshape((-1, 3))

                forces = (-gradient) * hartree_to_kcal / bohr_to_angstrom

                data_by_smiles[mapped_smiles].append({
                    "coords": coords,
                    "energy": energy,
                    "forces": forces,
                })

        # Optimization
        elif data_type == "optimization":
            last = record.trajectory[-1]
            last_mol = last.molecule
            mapped_smiles = last_mol.identifiers.canonical_isomeric_explicit_hydrogen_mapped_smiles
            if mapped_smiles is None:
                continue
            coords = last_mol.geometry * bohr_to_angstrom
            energy = last.properties["return_energy"] * hartree_to_kcal

            gradient = np.array(
                last.properties["scf total gradient"]
            ).reshape((-1, 3))

            forces = (-gradient) * hartree_to_kcal / bohr_to_angstrom

            data_by_smiles[mapped_smiles].append({
                "coords": coords,
                "energy": energy,
                "forces": forces,
            })        

    for mapped_smiles, entries in data_by_smiles.items():
        entry = {
            "smiles": mapped_smiles,
            "coords": torch.tensor([x["coords"] for x in entries]),
            "energy": torch.tensor([x["energy"] for x in entries]),
            "forces": torch.tensor([x["forces"] for x in entries]),
        }
        descent_entries.append(entry)

    dataset = descent.targets.energy.create_dataset(entries=descent_entries)
    dataset.set_format('torch', columns=['energy', 'coords','forces'], output_all_columns=True)    
    dataset.save_to_disk("test-smee-data")

    # print(dataset[0])

    return dataset

# Force Field Setup

def setup_forcefield(ff_path):
    ff = ForceField(
        ff_path,
        load_plugins=True,
        allow_cosmetic_attributes=True
    )
    return ff


def apply_functional_form(ff, form_name):
    print(form_name)
    if form_name == "LeeKrimm":
        ff.deregister_parameter_handler("ImproperTorsions")
    elif form_name == "HarmonicAngle":
        ff.deregister_parameter_handler("ImproperTorsions")
    elif form_name == "HarmonicHeight":
        ff.deregister_parameter_handler("ImproperTorsions")
    elif form_name == "TwoMinima":
        pass
    else:
        raise ValueError(f"Unknown functional form {form_name}")
    return ff


# Interchange + SMEE Conversion

def build_smee_system(dataset, forcefield):
    all_smiles = []
    interchanges = []

    for entry in tqdm.tqdm(dataset):
        mol = Molecule.from_mapped_smiles(entry["smiles"], allow_undefined_stereo=True)
        interchange_obj = forcefield.create_interchange(mol.to_topology())

        all_smiles.append(entry["smiles"])
        interchanges.append(interchange_obj)

    smee_ff, smee_topologies = smee.converters.convert_interchange(interchanges)
    topologies = dict(zip(all_smiles, smee_topologies))
    # print(smee_ff)
    return smee_ff, topologies


# Training Setup

def create_trainable(smee_force_field, functional_form):
    import math
    import descent.train

    parameters = {
        "Bonds": descent.train.ParameterConfig(
            cols=["k", "length"],
            scales={"k": 1e-2, "length": 1.0},
            limits={"k": [0.0, None], "length": [0.0, None]},
        ),
        "Angles": descent.train.ParameterConfig(
            cols=["k", "angle"],
            scales={"k": 1e-2, "angle": 1.0},
            limits={"k": [0.0, None], "angle": [0.0, math.pi]},
        ),
        "ProperTorsions": descent.train.ParameterConfig(
            cols=["k"],
            scales={"k": 1.0},
        ),
    }

    available = set(smee_force_field.potentials_by_type.keys())
    print("Available SMEE potentials:", available)

    if functional_form == "LeeKrimm":
        parameters["LeeKrimm"] = descent.train.ParameterConfig(
            cols=["V2", "V4", "t", "s"],
            scales={"V2": 1.0, "V4": 1.0},
            limits={
                "V2": [0.0, None],
                "V4": [0.0, None],
                "t": [0.0, None],
                "s": [0.0, None],
            },
        )

    elif functional_form == "HarmonicAngle":
        parameters["HarmonicAngle"] = descent.train.ParameterConfig(
            cols=["k", "theta0"],
            scales={"k": 1e-2, "theta0": 1.0},
            limits={"k": [0.0, None], "theta0": [0.0, math.pi]},
        )

    elif functional_form == "HarmonicHeight":
        parameters["HarmonicHeight"] = descent.train.ParameterConfig(
            cols=["k", "h0"],
            scales={"k": 1e-2, "h0": 1.0},
            limits={"k": [0.0, None], "h0": [0.0, None]},
        )

    elif functional_form == "TwoMinima":
        parameters["ImproperTorsions"] = descent.train.ParameterConfig(
            cols=["k"],
            scales={"k": 1.0},
        )

    else:
        raise ValueError(f"Unknown functional form: {functional_form}")

    return descent.train.Trainable(
        force_field=smee_force_field,
        parameters=parameters,
        attributes={},
    )

# Training Loop

def write_metrics(
        epoch: int,
        loss: torch.Tensor,
        loss_energy: torch.Tensor,
        loss_forces: torch.Tensor,
        writer: tensorboardX.SummaryWriter
):
    
    loss_val = loss.detach().item()
    energy_val = loss_energy.detach().item()
    force_val = loss_forces.detach().item()

    print(f"epoch={epoch} loss={loss.detach().item():.6f}", flush=True)

    writer.add_scalar("loss", loss.detach().item(), epoch)
    writer.add_scalar("loss_energy", loss_energy.detach().item(), epoch)
    writer.add_scalar("loss_forces", loss_forces.detach().item(), epoch)

    writer.add_scalar("rmse_energy", math.sqrt(loss_energy.detach().item()), epoch)
    writer.add_scalar("rmse_forces", math.sqrt(loss_forces.detach().item()), epoch)
    writer.flush()

    return loss_val, energy_val, force_val


import matplotlib.pyplot as plt

def save_curve(epoch_history, loss_history, energy_history, force_history, filename):

    plt.figure()

    plt.plot(epoch_history, loss_history, label="Total Loss")
    plt.plot(epoch_history, energy_history, label="Energy Loss")
    plt.plot(epoch_history, force_history, label="Force Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Curves")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def train_model(trainable, dataset, topologies, config, output_dir):
    epochs = config["epochs"]
    lr = config["learning_rate"]
    batch_size = config["batch_size"]

    trainable_parameters = trainable.to_values()
    device = trainable_parameters.device.type

    writer = tensorboardX.SummaryWriter(str(output_dir))
    optimizer = torch.optim.Adam([trainable_parameters], lr=lr, amsgrad=True)

    dataset_indices = list(range(len(dataset)))
    loss_history = []
    energy_history = []
    force_history = []
    epoch_history = []

    for i in range(epochs):
        ff = trainable.to_force_field(trainable_parameters)

        total_loss = torch.zeros(1, device=device)
        energy_loss = torch.zeros(1, device=device)
        force_loss = torch.zeros(1, device=device)

        grad = None

        for batch_ids in more_itertools.batched(dataset_indices, batch_size):
            batch = dataset.select(indices=batch_ids)
            true_batch_size = len(dataset)

            e_ref, e_pred, f_ref, f_pred = descent.targets.energy.predict(
                batch, ff, topologies, "mean"
            )

            loss_e = ((e_pred - e_ref) ** 2).sum() / true_batch_size
            loss_f = ((f_pred - f_ref) ** 2).sum() / true_batch_size
            loss = loss_e + loss_f

            (batch_grad,) = torch.autograd.grad(loss, trainable_parameters, create_graph=True)
            batch_grad = batch_grad.detach()

            grad = batch_grad if grad is None else grad + batch_grad

            total_loss += loss.detach()
            energy_loss += loss_e.detach()
            force_loss += loss_f.detach()

        trainable_parameters.grad = grad
                
        loss_val, energy_val, force_val = write_metrics(
            epoch=i,
            loss=total_loss,
            loss_energy=energy_loss,
            loss_forces=force_loss,
            writer=writer
        )

        epoch_history.append(i)
        loss_history.append(loss_val)
        energy_history.append(energy_val)
        force_history.append(force_val)

        print(f"Epoch {i} | Loss: {total_loss.item():.6f}")

        writer.add_scalar("loss", total_loss.item(), i)
        writer.flush()

        optimizer.step()
        optimizer.zero_grad()

        if i % 10 == 0:
            torch.save(
                trainable.to_force_field(trainable_parameters),
                output_dir / f"force-field-epoch-{i}.pt"
        )

    final_ff = trainable.to_force_field(trainable_parameters)
    torch.save(final_ff, output_dir / "final-force-field.pt")

    save_curve(
        epoch_history,
        loss_history,
        energy_history,
        force_history,
        output_dir / "training_loss.png"
    )
    
    return final_ff, trainable_parameters


from collections import defaultdict
import numpy as np


def update_forcefield_parameters(smee_force_field, trainable, base_forcefield):
    """
    Writes optimized SMEE parameters back into an OpenFF ForceField.

    Parameters
    ----------
    smee_force_field: smee.ForceField
    trainable: descent.train.Trainable
    base_forcefield: openff.toolkit.ForceField

    Returns
    -------
    updated_forcefield
    """

    for potential in smee_force_field.potentials:
        handler_name = potential.parameter_keys[0].associated_handler

        parameter_attrs = potential.parameter_cols
        parameter_units = potential.parameter_units

        print(f"Updating handler: {handler_name}")

        if handler_name in ["Bonds", "Angles"]:
            handler = base_forcefield.get_parameter_handler(handler_name)

            for i, opt_parameters in enumerate(potential.parameters):
                smirks = potential.parameter_keys[i].id
                ff_param = handler[smirks]

                values = opt_parameters.detach().cpu().numpy()

                for j, (attr, unit) in enumerate(zip(parameter_attrs, parameter_units)):
                    setattr(ff_param, attr, values[j] * unit)

        elif handler_name == "ProperTorsions":
            handler = base_forcefield.get_parameter_handler(handler_name)

            k_index = parameter_attrs.index("k")
            p_index = parameter_attrs.index("periodicity")

            collection = defaultdict(dict)

            for i, opt_parameters in enumerate(potential.parameters):
                smirks = potential.parameter_keys[i].id
                values = opt_parameters.detach().cpu().numpy()

                k = values[k_index] * parameter_units[k_index]
                p = int(values[p_index])

                collection[smirks][p] = k

            for smirks, k_map in collection.items():
                ff_param = handler[smirks]
                ff_param.k = [k_map[p] for p in ff_param.periodicity]

        elif handler_name == "ImproperTorsions":
            handler = base_forcefield.get_parameter_handler(handler_name)
            k_index = parameter_attrs.index('k')
            p_index = parameter_attrs.index('periodicity')
            # we need to collect the k values into a list across the entries
            collection_data = defaultdict(dict)
            for i, opt_parameters in enumerate(potential.parameters):
                smirks = potential.parameter_keys[i].id
                ff_parameter = handler[smirks]
                opt_parameters = opt_parameters.detach().cpu().numpy()
                # find k and the periodicity
                k = opt_parameters[k_index] * parameter_units[k_index]
                p = int(opt_parameters[p_index])
                collection_data[smirks][p] = k
            # now update the force field
            for smirks, k_s in collection_data.items():
                ff_parameter = handler[smirks]
                k_mapped_to_p = [k_s[p] for p in ff_parameter.periodicity]
                ff_parameter.k = k_mapped_to_p

        else:
            # This covers LeeKrimm, HarmonicAngle, HarmonicHeight, etc.
            if handler_name == "TwoMinima":
                continue
            else:
                try:
                    handler = base_forcefield.get_parameter_handler(handler_name)
                except Exception:
                    print(f"Skipping unknown handler: {handler_name}")
                    continue

                param_indices = {name: i for i, name in enumerate(parameter_attrs)}

                for i, opt_parameters in enumerate(potential.parameters):
                    smirks = potential.parameter_keys[i].id
                    ff_param = handler[smirks]

                    values = opt_parameters.detach().cpu().numpy()

                    for attr, unit in zip(parameter_attrs, parameter_units):
                        idx = param_indices[attr]
                        setattr(ff_param, attr, values[idx] * unit)

    return base_forcefield

# Output Utils

def create_output_dir(base, ff_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = pathlib.Path(base) / f"{ff_type}_run_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_forcefield(ff, output_dir, ff_type):
    ff.to_file(output_dir / f"{ff_type}_final.offxml")
