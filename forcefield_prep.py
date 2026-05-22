import json
from openff.toolkit import ForceField
from openff.units import unit


def get_or_create_handler(forcefield, handler_name):
    try:
        return forcefield.get_parameter_handler(handler_name)
    except Exception:
        return forcefield.add_parameter_handler(handler_name)



DEFAULT_SMIRKS = [
    "[*:1]~[#7X3:2](~[*:3])~[*:4]",
    "[*:1]~[#6X3:2](~[#8X1:3])~[#8:4]",
    "[*:1]~[#7X3$(*~[#15,#16](!-[*])):2](~[*:3])~[*:4]",
    "[*:1]~[#7X3$(*~[#6X3]):2](~[*:3])~[*:4]",
    "[*:1]~[#7X3$(*~[#7X2]):2](~[*:3])~[*:4]",
    "[*:1]~[#7X3$(*@1-[*]=,:[*][*]=,:[*]@1):2](~[*:3])~[*:4]",
    "[*:1]~[#6X3:2](=[#7X2,#7X3+1:3])~[#7:4]"
]


def add_lee_krimm(forcefield):
    for smirk in DEFAULT_SMIRKS:
        print(smirk)
        handler = get_or_create_handler(forcefield, "LeeKrimm")

        handler.add_parameter({
            "smirks": smirk,
            "V2": 5.0 * unit.kilojoule_per_mole,
            "V4": 5.0 * unit.kilojoule_per_mole,
            "t": 2.0,
            "s": 1.0,
        })


def add_harmonic_height(forcefield):
    for smirk in DEFAULT_SMIRKS:
        handler = get_or_create_handler(forcefield, "HarmonicHeight")

        handler.add_parameter({
            "smirks": smirk,
            "k": 100.0 * unit.kilojoule_per_mole * unit.nanometer ** -2,
            "h0": 1.0 * unit.nanometer,
        })


def add_harmonic_angle(forcefield):
    for smirk in DEFAULT_SMIRKS:
        handler = get_or_create_handler(forcefield, "HarmonicAngle")

        handler.add_parameter({
            "smirks": smirk,
            "k": 100.0 * unit.kilocalorie_per_mole * unit.radian ** -2,
            "theta0": 1.0 * unit.radian,
        })


def modify_two_minima(forcefield):

    handler = forcefield.get_parameter_handler("ImproperTorsions")

    for param in handler.parameters:
        if 4 not in param.periodicity:
            param.periodicity.append(4)
            param.phase.append(-180.0 * unit.degree)
            param.k.append(10.0 * unit.kilocalorie_per_mole)


def apply_functional_form(forcefield, form):

    if form == "LeeKrimm":
        add_lee_krimm(forcefield)

    elif form == "HarmonicHeight":
        add_harmonic_height(forcefield)

    elif form == "HarmonicAngle":
        add_harmonic_angle(forcefield)

    elif form == "TwoMinima":
        modify_two_minima(forcefield)

    else:
        raise ValueError(f"Unknown functional form: {form}")

    return forcefield



def main(config_path):

    with open(config_path) as f:
        config = json.load(f)

    ff = ForceField(
        config["base_forcefield"],
        load_plugins=True
    )

    ff = apply_functional_form(
        ff,
        config["functional_form"]
    )

    ff.to_file(f'{config["functional_form"]}_{config["output_forcefield"]}')

    print(f"Saved → {config['output_forcefield']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python forcefield_prep.py input.json")
        sys.exit(1)

    main(sys.argv[1])