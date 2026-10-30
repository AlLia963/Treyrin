"""Explicit one-token public alphabet, including all 9x5 control cells."""
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def public_label(index):
    if type(index) is not int or not 0 <= index < len(ALPHABET):
        raise ValueError('Reviewed public label alphabet exhausted')
    return ALPHABET[index]
