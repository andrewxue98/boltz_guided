from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


AtomContext = Mapping[str, int | str]
RESERVED_TOKENS = {"and", "or", "not", "to", "chain", "resid", "resi", "name", "atom", "index"}


class ParseError(ValueError):
    """Selection parsing error."""


class SelectionNode(Protocol):
    """Selection node protocol."""

    def evaluate(self, atom: AtomContext) -> bool:
        """Evaluate the node against an atom context."""


@dataclass(frozen=True)
class ChainSelection:
    names: tuple[str, ...]

    def evaluate(self, atom: AtomContext) -> bool:
        return str(atom.get("chain", "")).upper() in self.names


@dataclass(frozen=True)
class ResidSelection:
    values: tuple[int, ...]

    def evaluate(self, atom: AtomContext) -> bool:
        return int(atom.get("resid", -1)) in self.values


@dataclass(frozen=True)
class NameSelection:
    names: tuple[str, ...]

    def evaluate(self, atom: AtomContext) -> bool:
        return str(atom.get("name", "")).upper() in self.names


@dataclass(frozen=True)
class IndexSelection:
    values: tuple[int, ...]

    def evaluate(self, atom: AtomContext) -> bool:
        return int(atom.get("index", -1)) in self.values


@dataclass(frozen=True)
class NotSelection:
    node: SelectionNode

    def evaluate(self, atom: AtomContext) -> bool:
        return not self.node.evaluate(atom)


@dataclass(frozen=True)
class AndSelection:
    nodes: tuple[SelectionNode, ...]

    def evaluate(self, atom: AtomContext) -> bool:
        return all(node.evaluate(atom) for node in self.nodes)


@dataclass(frozen=True)
class OrSelection:
    nodes: tuple[SelectionNode, ...]

    def evaluate(self, atom: AtomContext) -> bool:
        return any(node.evaluate(atom) for node in self.nodes)


class SelectionParser:
    """Recursive-descent parser for a small PyMOL-like atom selector."""

    def __init__(self, text: str):
        self.tokens = self._tokenize(text)
        self.pos = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = []
        idx = 0
        while idx < len(text):
            char = text[idx]
            if char.isspace():
                idx += 1
                continue
            if char in {"(", ")"}:
                tokens.append(char)
                idx += 1
                continue
            if char.isdigit():
                end = idx + 1
                while end < len(text) and text[end].isdigit():
                    end += 1
                tokens.append(text[idx:end])
                idx = end
                continue
            if char.isalnum() or char == "_":
                end = idx + 1
                while end < len(text) and (text[end].isalnum() or text[end] == "_"):
                    end += 1
                tokens.append(text[idx:end])
                idx = end
                continue
            msg = f"Unexpected character {char!r} in selection."
            raise ParseError(msg)
        return tokens

    def _peek(self) -> str | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _peek_lower(self) -> str | None:
        token = self._peek()
        return None if token is None else token.lower()

    def _consume(self, expected: str | None = None) -> str:
        token = self._peek()
        if token is None:
            if expected is None:
                msg = "Unexpected end of selection."
            else:
                msg = f"Expected {expected!r}, found end of selection."
            raise ParseError(msg)
        if expected is not None and token.lower() != expected.lower():
            msg = f"Expected {expected!r}, found {token!r}."
            raise ParseError(msg)
        self.pos += 1
        return token

    def _consume_identifier(self) -> str:
        token = self._consume()
        if token in {"(", ")"} or token.isdigit():
            msg = f"Expected identifier, found {token!r}."
            raise ParseError(msg)
        if token.lower() in RESERVED_TOKENS:
            msg = f"Identifier cannot be reserved token {token!r}."
            raise ParseError(msg)
        return token.upper()

    def _consume_number(self) -> int:
        token = self._consume()
        if not token.isdigit():
            msg = f"Expected integer, found {token!r}."
            raise ParseError(msg)
        return int(token)

    def _parse_identifier_list(self) -> tuple[str, ...]:
        values = [self._consume_identifier()]
        while True:
            token = self._peek()
            if token is None or token in {"(", ")"} or token.lower() in RESERVED_TOKENS:
                break
            values.append(self._consume_identifier())
        return tuple(values)

    def _parse_number_list(self) -> tuple[int, ...]:
        start = self._consume_number()
        if self._peek_lower() == "to":
            self._consume("to")
            end = self._consume_number()
            if end < start:
                msg = f"Residue/index range end {end} cannot be smaller than start {start}."
                raise ParseError(msg)
            return tuple(range(start, end + 1))

        values = [start]
        while True:
            token = self._peek()
            if token is None or not token.isdigit():
                break
            values.append(self._consume_number())
        return tuple(values)

    def _parse_atom(self) -> SelectionNode:
        keyword = self._peek_lower()
        if keyword == "chain":
            self._consume("chain")
            return ChainSelection(self._parse_identifier_list())
        if keyword in {"resid", "resi"}:
            self._consume()
            return ResidSelection(self._parse_number_list())
        if keyword in {"name", "atom"}:
            self._consume()
            return NameSelection(self._parse_identifier_list())
        if keyword == "index":
            self._consume("index")
            return IndexSelection(self._parse_number_list())
        msg = f"Expected atomic selector at token {self._peek()!r}."
        raise ParseError(msg)

    def _parse_primary(self) -> SelectionNode:
        if self._peek() == "(":
            self._consume("(")
            node = self._parse_or()
            self._consume(")")
            return node
        return self._parse_atom()

    def _parse_not(self) -> SelectionNode:
        if self._peek_lower() == "not":
            self._consume("not")
            return NotSelection(self._parse_not())
        return self._parse_primary()

    def _parse_and(self) -> SelectionNode:
        nodes = [self._parse_not()]
        while self._peek_lower() == "and":
            self._consume("and")
            nodes.append(self._parse_not())
        if len(nodes) == 1:
            return nodes[0]
        return AndSelection(tuple(nodes))

    def _parse_or(self) -> SelectionNode:
        nodes = [self._parse_and()]
        while self._peek_lower() == "or":
            self._consume("or")
            nodes.append(self._parse_and())
        if len(nodes) == 1:
            return nodes[0]
        return OrSelection(tuple(nodes))

    def parse(self) -> SelectionNode:
        node = self._parse_or()
        if self._peek() is not None:
            msg = f"Unexpected trailing token {self._peek()!r}."
            raise ParseError(msg)
        return node


def parse_selection(text: str) -> SelectionNode:
    """Parse a selection string into a deterministic syntax tree."""

    return SelectionParser(text).parse()

