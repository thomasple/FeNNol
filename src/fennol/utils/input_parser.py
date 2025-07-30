"""
FeNNol Input Parameter Parser

This module provides parsing capabilities for FeNNol input files in .fnl format.
It supports hierarchical parameter organization, automatic unit conversion,
and various data types including booleans, numbers, strings, and lists.

The main components are:
- InputFile: Dictionary-like container for hierarchical parameters
- parse_input: Parser function for .fnl files  
- convert_dict_units: Unit conversion utility for YAML/dict inputs

Supported input formats:
- .fnl: Native FeNNol format with hierarchical sections
- .yaml/.yml: YAML format (processed through convert_dict_units)

Unit conversion:
- Units specified in brackets: dt[fs] = 0.5
- Units specified in braces: gamma{THz} = 10.0
- All units converted to atomic units internally

Boolean representations:
- True: true, yes, .true.
- False: false, no, .false.

"""

import re
from typing import Dict, Any
from .atomic_units import au,UnitSystem

_separators = " |,|=|\t|\n"
_comment_chars = ["#", "!"]
_true_repr = ["true", "yes", ".true."]
_false_repr = ["false", "no", ".false."]


class InputFile(dict):
    """
    Dictionary-like container for hierarchical input parameters.
    
    This class extends dict to provide path-based access to nested parameters
    using '/' as a separator. It supports case-insensitive keys and automatic
    unit conversion from parameter names with bracket notation.
    
    Attributes:
        case_insensitive (bool): Whether keys are case-insensitive (default: True)
    
    Examples:
        >>> params = InputFile()
        >>> params.store("xyz_input/file", "system.xyz")
        >>> params.get("xyz_input/file")
        'system.xyz'
        >>> params["temperature"] = 300.0
        >>> params.get("temperature")
        300.0
    """
    case_insensitive = True

    def __init__(self, *args, **kwargs):
        super(InputFile, self).__init__(*args, **kwargs)
        if InputFile.case_insensitive:
            for key in list(self.keys()):
                dict.__setitem__(self, key.lower(), dict.get(self, key))
        for key in list(self.keys()):
            if isinstance(self[key], dict):
                dict.__setitem__(self, key, InputFile(**self[key]))
        

    def get(self, path, default=None):
        if not isinstance(path, str):
            raise TypeError("Path must be a string")
        if InputFile.case_insensitive:
            path = path.lower()
        keys = path.split("/")
        val = None
        for key in keys:
            if isinstance(val, InputFile):
                val = val.get(key, default=None)
            else:
                val = dict.get(self, key, None)

            if val is None:
                return default

        return val

    def store(self, path, value):
        if not isinstance(path, str):
            raise TypeError("Path must be a string")
        if isinstance(value, dict):
            value = InputFile(**value)
        if InputFile.case_insensitive:
            path = path.lower()
        keys = path.split("/")
        child = self.get(keys[0], default=None)
        if isinstance(child, InputFile):
            if len(keys) == 1:
                print("Warning: overriding a sub-dictionary!")
                dict.__setitem__(self, keys[0], value)
                # self[keys[0]] = value
                return 1
            else:
                child.store("/".join(keys[1:]), value)
        else:
            if len(keys) == 1:
                dict.__setitem__(self, keys[0], value)
                # self[keys[0]] = value
                return 0
            else:
                if child is None:
                    sub_dict = InputFile()
                    sub_dict.store("/".join(keys[1:]), value)
                    dict.__setitem__(self, keys[0], sub_dict)
                else:
                    print("Error: hit a leaf before the end of path!")
                    return -1
    
    def __getitem__(self, path):
        return self.get(path)
    
    def __setitem__(self, path, value):
        return self.store(path, value)

    def print(self, tab=""):
        string = ""
        for p_id, p_info in self.items():
            string += tab + p_id
            val = self.get(p_id)
            if isinstance(val, InputFile):
                string += "{\n" + val.print(tab=tab + "  ") + "\n" + tab + "}\n\n"
            else:
                string += " = " + str(val) + "\n"
        return string[:-1]

    def save(self, filename):
        with open(filename, "w") as f:
            f.write(self.print())

    def __str__(self):
        return self.print()


def parse_input(input_file,us:UnitSystem=au):
    """
    Parse a FeNNol input file (.fnl format) into a hierarchical parameter structure.
    
    The parser supports:
    - Hierarchical sections using curly braces {}
    - Comments starting with # or !
    - Unit specifications in brackets [unit] => converted to provided UnitSystem (default: atomic units)
    - Boolean values (yes/no, true/false, .true./.false.)
    - Numeric values (int/float)
    - String values
    - Lists of values
    
    Parameters:
        input_file (str): Path to the input file
        
    Returns:
        InputFile: Hierarchical dictionary containing parsed parameters
        
    Example input file::
    
        device cuda:0
        temperature = 300.0
        dt[fs] = 0.5
        
        xyz_input{
            file system.xyz
            indexed yes
        }
        
        thermostat LGV
        gamma[THz] = 10.0
    """
    f = open(input_file, "r")
    struct = InputFile()
    path = []
    for line in f:
        # remove all after comment character
        for comment_char in _comment_chars:
            index = line.find(comment_char)
            if index >= 0:
                line = line[:index]

        # split line using defined separators
        parsed_line = re.split(_separators, line.strip())

        # remove empty strings
        parsed_line = [x for x in parsed_line if x]
        # skip blank lines
        if not parsed_line:
            continue
        # print(parsed_line)

        word0 = parsed_line[0].lower()
        cat_fields = "".join(parsed_line)
        # check if beginning of a category
        if cat_fields.endswith("{"):
            path.append(cat_fields[:-1])
            continue
        if cat_fields.startswith("&"):
            path.append(cat_fields[1:])
            continue
        if cat_fields.endswith("{}"):
            struct.store("/".join("path") + "/" + cat_fields[1:-2], InputFile())
            continue

        # print(current_category)
        # if not path:
        # 	print("Error: line not recognized!")
        # 	return None
        # else: #check if end of a category
        if (cat_fields[0] in "}/") or ("&end" in cat_fields):
            del path[-1]
            continue

        word0, unit = _get_unit_from_key(word0,us)
        val = None
        if len(parsed_line) == 1:
            val = True  # keyword only => store True
        elif len(parsed_line) == 2:
            val = string_to_true_type(parsed_line[1], unit)
        else:
            # analyze parsed line
            val = []
            for word in parsed_line[1:]:
                val.append(string_to_true_type(word, unit))
        struct.store("/".join(path + [word0]), val)

    f.close()
    return struct


def string_to_true_type(word, unit=None):
    if unit is not None:
        return float(word) / unit

    try:
        val = int(word)
    except ValueError:
        try:
            val = float(word)
        except ValueError:
            if word.lower() in _true_repr:
                val = True
            elif word.lower() in _false_repr:
                val = False
            else:
                val = word
    return val


def _get_unit_from_key(word:str,us:UnitSystem):
    unit_start = max(word.find("{"), word.find("["))
    n = len(word)
    if unit_start < 0:
        key = word
        unit = None
    elif unit_start == 0:
        print("Error: Field '" + str(word) + "' must not start with '{' or '[' !")
        raise ValueError
    else:
        if word[unit_start] == "{":
            end_bracket = "}"
        else:
            end_bracket = "]"
        key = word[:unit_start]
        if word[n - 1] != end_bracket:
            print("Error: wrong unit specification in field '" + str(word) + "' !")
            raise ValueError

        if n - unit_start - 2 < 0:
            unit = 1.0
        else:
            unit = us.get_multiplier(word[unit_start + 1 : -1])
            # print(key+" unit= "+str(unit))
    return key, unit


def convert_dict_units(d: Dict[str, Any],us:UnitSystem=au) -> Dict[str, Any]:
    """
    Convert all values in a dictionary from specified units to the provided unit system (atomic units by default).
    
    This function recursively processes a dictionary and converts any values
    with unit specifications (indicated by keys containing [unit] or {unit})
    to atomic units. The unit specification is removed from the key name.
    
    Parameters:
        d (Dict[str, Any]): Dictionary with potentially unit-specified keys
        
    Returns:
        Dict[str, Any]: Dictionary with values converted to atomic units
        
    Examples:
        >>> d = {"dt[fs]": 0.5, "temperature": 300.0}
        >>> convert_dict_units(d)
        {"dt": 20.67..., "temperature": 300.0}
    """

    if not isinstance(d, dict):
        raise TypeError("Input must be a dictionary")
    if not d:
        return d
    if not isinstance(us, UnitSystem):
        raise TypeError("Unit system must be an instance of UnitSystem")
    d2 = {}
    for k,v in d.items():
        if isinstance(v, dict):
            d2[k] = convert_dict_units(v,us)
            continue
        key, unit = _get_unit_from_key(k,us)
        if unit is None:
            d2[k] = v
            continue
        try:
            if isinstance(v, list):
                d2[key] = [x / unit for x in v]
            elif isinstance(v, tuple):
                d2[key] = tuple(x / unit for x in v)
            else:
                d2[key] = v / unit
        except TypeError:
            raise ValueError(f"Error: cannot convert value '{v}' of type to atomic units.")
        except Exception as e:
            print(f"Error: unexpected error in unit conversion for key '{k}': {e}")
            raise e

    return d2
