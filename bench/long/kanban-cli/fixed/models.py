"""Data models for the kan kanban board CLI.

Defines Board, Column, and Card dataclasses along with validation rules for
names, tags, and state transitions.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Validation constants
MAX_NAME_LENGTH = 100
MAX_TAG_LENGTH = 40
MAX_TAGS_PER_CARD = 10
MAX_TITLE_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 2000
MAX_COLUMNS = 20
MAX_CARDS_PER_COLUMN = 1000

# Default column names for a new board
DEFAULT_COLUMNS = ["Todo", "In Progress", "Done"]

# A tag is a short alphanumeric token that may contain hyphens/underscores.
_TAG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


class ValidationError(ValueError):
    """Raised when a model fails validation."""


def _new_id() -> str:
    """Generate a short unique id for a card."""
    return uuid.uuid4().hex[:8]


def validate_name(name: str, *, field_name: str = "name") -> str:
    """Validate a board/column/card name. Returns the stripped name."""
    if not isinstance(name, str):
        raise ValidationError(f"{field_name} must be a string")
    stripped = name.strip()
    if not stripped:
        raise ValidationError(f"{field_name} must not be empty")
    if len(stripped) > MAX_NAME_LENGTH:
        raise ValidationError(
            f"{field_name} must not exceed {MAX_NAME_LENGTH} characters"
        )
    return stripped


def validate_title(title: str) -> str:
    """Validate a card title. Returns the stripped title."""
    if not isinstance(title, str):
        raise ValidationError("title must be a string")
    stripped = title.strip()
    if not stripped:
        raise ValidationError("title must not be empty")
    if len(stripped) > MAX_TITLE_LENGTH:
        raise ValidationError(
            f"title must not exceed {MAX_TITLE_LENGTH} characters"
        )
    return stripped


def validate_description(description: Optional[str]) -> str:
    """Validate a card description. Returns the stripped description or ''."""
    if description is None:
        return ""
    if not isinstance(description, str):
        raise ValidationError("description must be a string")
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise ValidationError(
            f"description must not exceed {MAX_DESCRIPTION_LENGTH} characters"
        )
    return description


def validate_tag(tag: str) -> str:
    """Validate a single tag. Returns the stripped, lowercased tag."""
    if not isinstance(tag, str):
        raise ValidationError("tag must be a string")
    stripped = tag.strip().lower()
    if not stripped:
        raise ValidationError("tag must not be empty")
    if len(stripped) > MAX_TAG_LENGTH:
        raise ValidationError(f"tag must not exceed {MAX_TAG_LENGTH} characters")
    if not _TAG_RE.match(stripped):
        raise ValidationError(
            f"tag '{stripped}' must be alphanumeric (hyphens/underscores allowed)"
        )
    return stripped


def validate_tags(tags: Optional[List[str]]) -> List[str]:
    """Validate a list of tags. Returns a deduplicated, sorted list."""
    if tags is None:
        return []
    if not isinstance(tags, list):
        raise ValidationError("tags must be a list")
    cleaned: List[str] = []
    seen = set()
    for tag in tags:
        normalized = validate_tag(tag)
        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    if len(cleaned) > MAX_TAGS_PER_CARD:
        raise ValidationError(
            f"a card may have at most {MAX_TAGS_PER_CARD} tags"
        )
    return sorted(cleaned)


@dataclass
class Card:
    """A single kanban card."""

    title: str
    id: str = field(default_factory=_new_id)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    column: str = "Todo"

    def __post_init__(self) -> None:
        self.title = validate_title(self.title)
        self.description = validate_description(self.description)
        self.tags = validate_tags(self.tags)
        self.column = validate_name(self.column, field_name="column")

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
            "column": self.column,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Card":
        try:
            return cls(
                title=data["title"],
                id=data.get("id") or _new_id(),
                description=data.get("description", ""),
                tags=data.get("tags", []),
                column=data.get("column", "Todo"),
            )
        except KeyError as exc:
            raise ValidationError(f"card missing required field: {exc}")

    def has_tag(self, tag: str) -> bool:
        return validate_tag(tag) in self.tags


@dataclass
class Column:
    """A named column on the board that holds cards."""

    name: str
    cards: List[Card] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = validate_name(self.name, field_name="column name")

    def add_card(self, card: Card) -> None:
        if len(self.cards) >= MAX_CARDS_PER_COLUMN:
            raise ValidationError(
                f"column '{self.name}' is full "
                f"(max {MAX_CARDS_PER_COLUMN} cards)"
            )
        card.column = self.name
        self.cards.append(card)

    def remove_card(self, card_id: str) -> Optional[Card]:
        for i, card in enumerate(self.cards):
            if card.id == card_id:
                return self.cards.pop(i)
        return None

    def find_card(self, identifier: str) -> Optional[Card]:
        """Find a card by id or title (case-insensitive title match)."""
        for card in self.cards:
            if card.id == identifier or card.title.lower() == identifier.lower():
                return card
        return None

    def to_dict(self) -> Dict:
        return {"name": self.name, "cards": [c.to_dict() for c in self.cards]}

    @classmethod
    def from_dict(cls, data: Dict) -> "Column":
        try:
            name = data["name"]
        except KeyError as exc:
            raise ValidationError(f"column missing required field: {exc}")
        cards = [Card.from_dict(c) for c in data.get("cards", [])]
        return cls(name=name, cards=cards)


@dataclass
class Board:
    """A kanban board composed of ordered columns."""

    name: str = "My Board"
    columns: List[Column] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = validate_name(self.name, field_name="board name")
        if len(self.columns) > MAX_COLUMNS:
            raise ValidationError(
                f"board may have at most {MAX_COLUMNS} columns"
            )
        # Ensure column names are unique (case-insensitive).
        seen = set()
        for col in self.columns:
            key = col.name.lower()
            if key in seen:
                raise ValidationError(f"duplicate column name: {col.name}")
            seen.add(key)

    @classmethod
    def create_default(cls, name: str = "My Board") -> "Board":
        """Create a board with the default Todo/In Progress/Done columns."""
        columns = [Column(name=n) for n in DEFAULT_COLUMNS]
        return cls(name=name, columns=columns)

    def add_column(self, column: Column) -> None:
        if len(self.columns) >= MAX_COLUMNS:
            raise ValidationError(
                f"board may have at most {MAX_COLUMNS} columns"
            )
        if any(c.name.lower() == column.name.lower() for c in self.columns):
            raise ValidationError(f"duplicate column name: {column.name}")
        self.columns.append(column)

    def get_column(self, name: str) -> Optional[Column]:
        target = name.strip().lower()
        for col in self.columns:
            if col.name.lower() == target:
                return col
        return None

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def find_card(self, identifier: str) -> tuple:
        """Find a card anywhere on the board by id or title.

        Returns (column, card) or (None, None).
        """
        for col in self.columns:
            card = col.find_card(identifier)
            if card is not None:
                return col, card
        return None, None

    def all_cards(self) -> List[Card]:
        cards: List[Card] = []
        for col in self.columns:
            cards.extend(col.cards)
        return cards

    def add_card(self, title: str, column: str, description: str = "",
                 tags: Optional[List[str]] = None) -> Card:
        """Create and add a card to the named column."""
        col = self.get_column(column)
        if col is None:
            raise ValidationError(f"unknown column: {column}")
        card = Card(title=title, description=description, tags=tags or [])
        col.add_card(card)
        return card

    def move_card(self, identifier: str, target_column: str) -> Card:
        """Move a card (by id or title) to the target column.

        Validates the state transition: a card may not move from Done back to
        Todo or In Progress without first being explicitly allowed via the
        ``allow_done_revert`` flag. Returns the moved card.
        """
        target = self.get_column(target_column)
        if target is None:
            raise ValidationError(f"unknown column: {target_column}")
        source, card = self.find_card(identifier)
        if card is None:
            raise ValidationError(f"card not found: {identifier}")

        # State transition validation.
        self._validate_transition(card.column, target.name)
        source.remove_card(card.id)
        target.add_card(card)
        return card

    @staticmethod
    def _validate_transition(from_col: str, to_col: str) -> None:
        """Validate a state transition between columns.

        The only forbidden transition is moving *backwards* out of a column
        named 'Done' into 'Todo' or 'In Progress'. Moving within the same
        column is a no-op and allowed.
        """
        if from_col == to_col:
            return
        from_norm = from_col.strip().lower()
        to_norm = to_col.strip().lower()
        if from_norm == "done" and to_norm in ("todo", "in progress"):
            raise ValidationError(
                "cannot move a card from 'Done' back to a non-Done column"
            )

    def delete_card(self, identifier: str) -> Card:
        """Remove and return a card by id or title."""
        source, card = self.find_card(identifier)
        if card is None:
            raise ValidationError(f"card not found: {identifier}")
        source.remove_card(card.id)
        return card

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Board":
        try:
            name = data["name"]
        except KeyError as exc:
            raise ValidationError(f"board missing required field: {exc}")
        columns = [Column.from_dict(c) for c in data.get("columns", [])]
        return cls(name=name, columns=columns)
