from typing import List, _LiteralGenericAlias, get_args, Tuple

Triple = Tuple[str, str, str]

def get_list_from_literal(literal: _LiteralGenericAlias) -> List[str]:
    """
    Get a list of strings from a Literal type.

    Parameters:
    literal (_LiteralGenericAlias): The Literal type from which to extract the strings.

    Returns:
    List[str]: A list of strings extracted from the Literal type.
    """
    if not isinstance(literal, _LiteralGenericAlias):
        raise TypeError(
            f"{literal} must be a Literal type.\nTry using typing.Literal{literal}."
        )
    return list(get_args(literal))


def remove_empty_values(input_dict):
    """
    Remove entries with empty values from the dictionary.

    Parameters:
    input_dict (dict): The dictionary from which empty values need to be removed.

    Returns:
    dict: A new dictionary with all empty values removed.
    """
    # Create a new dictionary excluding empty values and remove the `e.` prefix from the keys
    return {key.replace("e.", ""): value for key, value in input_dict.items() if value}


def get_filtered_props(records: dict, filter_list: List[str]) -> dict:
    return {k: v for k, v in records.items() if k not in filter_list}


# Lookup entry by middle value of tuple
def lookup_relation(relation: str, triples: List[Triple]) -> Triple:
    """
    Look up a triple in a list of triples by the middle value.
    """
    for triple in triples:
        if triple[1] == relation:
            return triple
    return None