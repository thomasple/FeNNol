from numpy import pi


class AtomicUnits:
    EV = 27.211386024367243  # Hartree to eV
    KCALPERMOL = 627.5096080305927  # Hartree to kcal/mol
    KJPERMOL = 2625.5002  # Hartree to kJ/mol
    BOHR = 0.52917721  # Bohr to Angstrom
    MPROT = 1836.1526734252586  # proton mass
    HBAR = 1.0  # Planck's constant
    FS = 2.4188843e-2  # AU time to femtoseconds
    PS = FS / 1000  # AU time to picoseconds
    KELVIN = 3.15774e5  # Hartree to Kelvin
    THZ = 1000.0 / FS  # AU frequency to THz
    NNEWTON = 82.387  # AU Force to nNewton
    CM1 = 219471.52  # Hartree to cm-1
    GMOL_AFS = BOHR / (MPROT * FS)  # AU momentum to (g/mol).A/fs
    KBAR = 294210.2648438959  # Hartree/bohr**3 to kbar
    ATM = KBAR * 1000.0 / 1.01325  # Hartree/bohr**3 to atm
    GPA = 0.1 * KBAR  # Hartree/bohr**3 to GPa
    DEBYE = 2.541746  # e.Bohr to Debye
    FSC = 1.0 / 137.035999084  # Fine structure constant
    MOL = 6.02214129e+23
    KCAL = KCALPERMOL * MOL
    KJ = KJPERMOL * MOL
    RY = 2.  # Hartree to Rydberg

    mapping = {
        "1": 1.0,
        "AU": 1.0,
        "HA":1.0,
        "EV": EV,
        "MEV": 1.0e3 * EV, # milli electronvolt
        "KCALPERMOL": KCALPERMOL,
        "KJPERMOL": KJPERMOL,
        "ANGSTROM": BOHR,
        "BOHR": 1.0 / BOHR,
        "AMU": 1.0 / MPROT,
        "FEMTOSECONDS": FS,
        "FS": FS,
        "PICOSECONDS": FS / 1000.0,
        "PS": FS / 1000.0,
        "KELVIN": KELVIN,
        "K": KELVIN,
        "THZ": THZ,
        "TRADHZ": THZ / (2 * pi),
        "CM-1": CM1,
        "CM1": CM1,
        "GMOLAFS": GMOL_AFS,
        "KBAR": KBAR,
        "ATM": ATM,
        "GPA": GPA,
        "DEBYE": DEBYE,
        "FSC": FSC,
        "MOL": MOL,
        "MOLE": MOL,
        "KCAL": KCAL,
        "KJ": KJ,
        "RY": RY,
    }

    @staticmethod
    def get_multiplier(unit_string):
        unit_string = unit_string.upper().strip()

        multiplier = 1.0

        unit_start = 0
        unit_stop = 0
        in_power = False
        power_start = 0
        power_stop = 0
        tmp_power = 1.0
        for i in range(len(unit_string)):
            current_char = unit_string[i]
            # print(current_char)
            if current_char == "^":
                if in_power:
                    print("Error: Syntax error in unit '" + unit_string + "' !")
                    raise ValueError

                in_power = True
                unit_stop = i - 1
                if unit_stop - unit_start < 0:
                    print("Error: Syntax error in unit '" + unit_string + "' !")
                    raise ValueError

            elif current_char == "{":
                if not in_power:
                    print("Error: Syntax error in unit '" + unit_string + "' !")
                    raise ValueError

                if i + 1 >= len(unit_string):
                    print("Error: Syntax error in unit '" + unit_string + "' !")
                    raise ValueError

                power_start = i + 1

            elif current_char == "}":
                if not in_power:
                    print("Error: Syntax error in unit '" + unit_string + "' !")
                    raise ValueError

                in_power = False
                power_stop = i - 1

                if power_stop - power_start < 0:
                    print("Error: Syntax error in unit '" + unit_string + "' !")
                    raise ValueError
                else:
                    tmp_power_read = int(unit_string[power_start : power_stop + 1])
                    tmp_power = tmp_power * tmp_power_read

                unit_substring = unit_string[unit_start : unit_stop + 1]
                tmp_unit = AtomicUnits.mapping[unit_substring]

                multiplier = multiplier * (tmp_unit**tmp_power)

                power_start = 0
                power_stop = 0
                unit_start = i + 1
                unit_stop = 0

            elif current_char == "*":
                if in_power:
                    print("Error: Syntax error in unit '" + unit_string + "' !")
                    raise ValueError

                if unit_start == i:
                    unit_start = i + 1
                    tmp_power = 1.0
                    continue

                unit_stop = i - 1
                if unit_stop - unit_start < 0:
                    print("Error: Syntax error in unit '" + unit_string + "' !")
                    raise ValueError

                unit_substring = unit_string[unit_start : unit_stop + 1]
                tmp_unit = AtomicUnits.mapping[unit_substring]
                multiplier = multiplier * (tmp_unit**tmp_power)

                unit_start = i + 1
                unit_stop = 0
                tmp_power = 1.0

            elif current_char == "/":
                if in_power:
                    print("Error: Syntax error in unit '" + unit_string + "' !")
                    raise ValueError

                if unit_start == i:
                    unit_start = i + 1
                    tmp_power = -1.0
                    continue

                unit_stop = i - 1
                if unit_stop - unit_start < 0:
                    print("Error: Syntax error in unit '" + unit_string + "' !")
                    raise ValueError

                unit_substring = unit_string[unit_start : unit_stop + 1]
                tmp_unit = AtomicUnits.mapping[unit_substring]
                multiplier = multiplier * (tmp_unit**tmp_power)

                unit_start = i + 1
                unit_stop = 0
                tmp_power = -1.0

            else:
                if i + 1 >= len(unit_string):
                    if in_power:
                        print("Error: Syntax error in unit '" + unit_string + "' !")
                        raise ValueError

                    unit_stop = i
                    if unit_stop - unit_start < 0:
                        print("Error: Syntax error in unit '" + unit_string + "' !")
                        raise ValueError

                    unit_substring = unit_string[unit_start : unit_stop + 1]
                    tmp_unit = AtomicUnits.mapping[unit_substring]
                    multiplier = multiplier * (tmp_unit**tmp_power)

        return multiplier
