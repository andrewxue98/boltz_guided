import pytest

from boltz.data.parse.selection import ParseError, parse_selection


def test_selection_parser_supports_boolean_combinations():
    selection = parse_selection("(chain A and resid 10 to 12 and name CA CB) or not chain B")

    assert selection.evaluate({"chain": "A", "resid": 11, "name": "CA", "index": 1})
    assert selection.evaluate({"chain": "C", "resid": 2, "name": "NZ", "index": 7})
    assert not selection.evaluate(
        {"chain": "B", "resid": 11, "name": "CA", "index": 1}
    )


def test_selection_parser_supports_aliases_and_one_based_index():
    selection = parse_selection("chain a and resi 4 and atom nz and index 2")

    assert selection.evaluate({"chain": "A", "resid": 4, "name": "NZ", "index": 2})
    assert not selection.evaluate(
        {"chain": "A", "resid": 4, "name": "NZ", "index": 1}
    )


def test_selection_parser_rejects_invalid_syntax():
    with pytest.raises(ParseError):
        parse_selection("chain A and (resid 1 or")

    with pytest.raises(ParseError):
        parse_selection("chain A resid 1")
