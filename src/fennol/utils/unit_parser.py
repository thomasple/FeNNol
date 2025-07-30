from math import pi
import math
import re
import copy
import copy
from typing import Dict, Union, Optional


def _tokenize_unit_string(unit_string: str) -> list:
    """
    Tokenize a unit string into components.
    
    Returns a list of tokens where each token is either:
    - A unit name (string)
    - An operator ('*', '/', '^')
    - A power specification (string starting with number or '{' or '(')
    - Parentheses ('(', ')')
    """
    # Define simpler regex patterns
    unit_name_pattern = r'[A-Z0-9_-]+'
    braced_power_pattern = r'\{[^}]*\}'
    numeric_power_pattern = r'[+-]?\d+(?:\.\d+)?'
    parenthesized_power_pattern = r'\([^\)]*\)'
    
    tokens = []
    pos = 0
    
    while pos < len(unit_string):
        # Skip whitespace
        while pos < len(unit_string) and unit_string[pos].isspace():
            pos += 1
        
        if pos >= len(unit_string):
            break
        
        current_char = unit_string[pos]
        
        # Handle parentheses
        if current_char in '()':
            tokens.append(current_char)
            pos += 1
            continue
        
        # Handle operators
        if current_char in '*/':
            tokens.append(current_char)
            pos += 1
            continue
        
        # Handle power operator
        if current_char == '^':
            tokens.append(current_char)
            pos += 1
            
            # Skip whitespace after ^
            while pos < len(unit_string) and unit_string[pos].isspace():
                pos += 1
            
            if pos >= len(unit_string):
                raise ValueError("Power operator '^' not followed by power value")
            
            # Look for braced power: {2}, {-3}, {2.5}
            braced_match = re.match(braced_power_pattern, unit_string[pos:])
            if braced_match:
                tokens.append(braced_match.group(0)[1:-1].strip())  # Remove braces
                pos += braced_match.end()
                continue

            # Look for parenthesized power: (2), (-3), (2.5)
            parenthesized_match = re.match(parenthesized_power_pattern, unit_string[pos:])
            if parenthesized_match:
                tokens.append(parenthesized_match.group(0)[1:-1].strip())
                pos += parenthesized_match.end()
                continue
            
            # Look for numeric power: 2, -3, 2.5
            numeric_match = re.match(numeric_power_pattern, unit_string[pos:])
            if numeric_match:
                tokens.append(numeric_match.group(0).strip())
                pos += numeric_match.end()
                continue
            
            # No valid power found
            raise ValueError(f"Invalid power specification starting at position {pos}")
        
        # Handle unit names
        unit_match = re.match(unit_name_pattern, unit_string[pos:])
        if unit_match:
            tokens.append(unit_match.group(0))
            pos += unit_match.end()
            continue
        
        # If we get here, we have an invalid character
        raise ValueError(f"Invalid character '{current_char}' at position {pos} in unit string '{unit_string}'")
    
    return tokens


def _validate_syntax(tokens: list) -> None:
    """
    Validate the syntax of tokenized unit string.
    """
    if not tokens:
        raise ValueError("Empty unit string")
    
    # Check for invalid starting/ending tokens
    if tokens[0] in ['*', '/', '^']:
        raise ValueError(f"Unit string cannot start with operator '{tokens[0]}'")
    if tokens[-1] in ['*', '/', '^']:
        raise ValueError(f"Unit string cannot end with operator '{tokens[-1]}'")
    
    # Check for balanced parentheses
    paren_count = 0
    for i, token in enumerate(tokens):
        if token == '(':
            paren_count += 1
        elif token == ')':
            paren_count -= 1
            if paren_count < 0:
                raise ValueError(f"Unmatched closing parenthesis at position {i}")
    
    if paren_count > 0:
        raise ValueError("Unmatched opening parenthesis")
    
    # Check for invalid consecutive operators
    for i in range(len(tokens) - 1):
        current, next_token = tokens[i], tokens[i + 1]
        
        # Check for consecutive operators (except ^ followed by power)
        if current in ['*', '/'] and next_token in ['*', '/', '^']:
            raise ValueError(f"Invalid consecutive operators: '{current}' followed by '{next_token}'")
        
        # Check for ^ not followed by power
        if current == '^' and not (next_token.startswith('{') or next_token.replace('.', '').replace('-', '').replace('+', '').isdigit()):
            raise ValueError(f"Power operator '^' must be followed by a number, got '{next_token}'")
        
        # Check for empty parentheses
        if current == '(' and next_token == ')':
            raise ValueError("Empty parentheses are not allowed")
        
        # Check for invalid parentheses placement
        if current == '(' and next_token in ['*', '/', '^']:
            raise ValueError(f"Opening parenthesis cannot be followed by operator '{next_token}'")
        if current in ['*', '/', '^'] and next_token == ')':
            raise ValueError(f"Operator '{current}' cannot be followed by closing parenthesis")


def _find_matching_paren(tokens: list, start_pos: int) -> int:
    """
    Find the position of the matching closing parenthesis for the opening parenthesis at start_pos.
    """
    if tokens[start_pos] != '(':
        raise ValueError("Expected opening parenthesis")
    
    paren_count = 1
    pos = start_pos + 1
    
    while pos < len(tokens) and paren_count > 0:
        if tokens[pos] == '(':
            paren_count += 1
        elif tokens[pos] == ')':
            paren_count -= 1
        pos += 1
    
    if paren_count > 0:
        raise ValueError("Unmatched opening parenthesis")
    
    return pos - 1  # Return position of closing parenthesis


def _parse_expression(tokens: list, unit_dict: Dict[str, float]) -> float:
    """
    Parse a list of tokens and return the multiplier.
    This is a recursive descent parser that handles parentheses.
    
    Args:
        tokens: List of tokens to parse
        unit_dict: Dictionary mapping unit names to their multipliers
    
    Returns:
        float: The calculated multiplier
    """
    if not tokens:
        raise ValueError("Empty expression")
    
    # Handle parentheses by recursively parsing sub-expressions
    processed_tokens = []
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        
        if token == '(':
            # Find matching closing parenthesis
            close_pos = _find_matching_paren(tokens, i)
            
            # Recursively parse the sub-expression inside parentheses
            sub_tokens = tokens[i+1:close_pos]
            sub_result = _parse_expression(sub_tokens, unit_dict)
            
            # Replace the parenthesized expression with its result
            # Create a synthetic unit name to represent the result
            synthetic_unit = f"__RESULT_{len(processed_tokens)}__"
            unit_dict[synthetic_unit] = sub_result
            processed_tokens.append(synthetic_unit)
            
            i = close_pos + 1
            continue
        
        elif token == ')':
            # This should not happen if validation passed
            raise ValueError("Unexpected closing parenthesis")
        
        else:
            processed_tokens.append(token)
            i += 1
    
    # Now parse the processed tokens without parentheses
    return _parse_linear_expression(processed_tokens, unit_dict)


def _parse_linear_expression(tokens: list, unit_dict: Dict[str, float]) -> float:
    """
    Parse a linear expression (no parentheses) and return the multiplier.
    
    Args:
        tokens: List of tokens to parse
        unit_dict: Dictionary mapping unit names to their multipliers
    
    Returns:
        float: The calculated multiplier
    """
    multiplier = 1.0
    current_sign = 1.0  # 1.0 for multiplication, -1.0 for division
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        if token in ['*', '/']:
            # Set sign for next unit
            current_sign = 1.0 if token == '*' else -1.0
            i += 1
            continue
        
        elif token == '^':
            # This should not happen if validation passed
            raise ValueError("Unexpected '^' token")
        
        else:
            # This should be a unit name
            unit_name = token
            power = current_sign  # Default power is 1.0 or -1.0 based on sign
            
            # Check if next token is a power specification
            if i + 1 < len(tokens) and tokens[i + 1] == '^':
                if i + 2 >= len(tokens):
                    raise ValueError("Power operator '^' not followed by power value")
                
                power_spec = tokens[i + 2]
                try:
                    power_value = float(power_spec)
                except ValueError:
                    raise ValueError(f"Invalid power specification: '{power_spec}'")
                power = current_sign * power_value
                i += 2  # Skip the '^' and power tokens
            
            # Look up unit multiplier
            if unit_name not in unit_dict:
                raise ValueError(f"Unknown unit: {unit_name}")
            
            unit_multiplier = unit_dict[unit_name]
            multiplier *= unit_multiplier ** power
            
            # Reset sign for next unit
            current_sign = 1.0
        
        i += 1
    
    return multiplier


def parse_unit_string(unit_string: str, unit_dict: Dict[str, float]) -> float:
    """
    Parse a unit string and return the conversion multiplier.
    
    This is the main entry point for unit string parsing. It handles the complete
    parsing pipeline: tokenization, validation, and calculation.
    
    Args:
        unit_string: The unit string to parse (e.g., "EV*ANGSTROM^2")
        unit_dict: Dictionary mapping unit names to their multipliers
    
    Returns:
        float: The calculated multiplier
    
    Raises:
        ValueError: If the unit string is invalid or contains unknown units
    
    Example:
        >>> unit_dict = {"EV": 27.211, "ANGSTROM": 0.529}
        >>> parse_unit_string("EV*ANGSTROM^2", unit_dict)
        7.619964
    """
    unit_string = unit_string.upper().strip()
    
    if not unit_string:
        raise ValueError("Empty unit string")
    
    # Create a deep copy of the unit dictionary to avoid modification
    unit_dict_copy = copy.deepcopy(unit_dict)
    
    # Tokenize the unit string
    try:
        tokens = _tokenize_unit_string(unit_string)
    except ValueError as e:
        raise ValueError(f"Syntax error in unit '{unit_string}': {str(e)}")
    
    # Validate syntax
    _validate_syntax(tokens)
    
    # Parse tokens to calculate multiplier using recursive descent parser
    try:
        return _parse_expression(tokens, unit_dict_copy)
    except ValueError as e:
        raise ValueError(f"Error parsing unit '{unit_string}': {str(e)}")