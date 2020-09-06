from typing import Any, List


def pad_array(in_list: List[Any], max_length: int, value: Any = 0) -> List[Any]:
    """Pad a list to the given size with the given padding value.

    # Arguments:
    in_list (List[Any]): List of values to pad to the given length
    max_length (int): The desired length of the array
    value (Any): a value to insert in order to pad the array to max length

    # Returns
    (List[Any]): An array padded to `max_length` size
    """
    while len(in_list) < max_length:
        in_list.append(value)
    return in_list
