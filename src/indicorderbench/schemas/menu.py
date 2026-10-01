"""Menu schema: items, modifier groups, alias lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ID_PATTERN = r"^[a-z][a-z0-9_]*$"

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace."""
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower())).strip()


@dataclass(frozen=True)
class AliasEntry:
    alias: str  # normalised
    kind: Literal["item", "option"]
    ref_id: str


class Modifier(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=ID_PATTERN)
    name: str
    aliases: list[str] = Field(default_factory=list)
    price_delta: Decimal = Decimal("0")


class ModifierGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=ID_PATTERN)
    name: str
    exclusive: bool = True
    default: str | None = None
    options: list[Modifier] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> ModifierGroup:
        ids = [o.id for o in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate option ids in group {self.id}")
        if self.exclusive:
            if self.default is None or self.default not in ids:
                raise ValueError(
                    f"exclusive group {self.id} needs a default that is one of its options"
                )
        elif self.default is not None:
            raise ValueError(f"non-exclusive group {self.id} must not declare a default")
        return self

    def option(self, option_id: str) -> Modifier:
        for o in self.options:
            if o.id == option_id:
                return o
        raise KeyError(option_id)


class MenuItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=ID_PATTERN)
    name: str
    aliases: list[str] = Field(default_factory=list)
    price: Decimal
    category: str = "general"
    modifier_groups: list[str] = Field(default_factory=list)


class Menu(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=ID_PATTERN)
    name: str
    currency: str = "INR"
    modifier_groups: list[ModifierGroup] = Field(default_factory=list)
    items: list[MenuItem] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Menu:
        group_ids = [g.id for g in self.modifier_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("duplicate modifier group ids")
        option_ids = [o.id for g in self.modifier_groups for o in g.options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("option ids must be unique across groups")
        item_ids = [i.id for i in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("duplicate item ids")
        if set(item_ids) & set(option_ids):
            raise ValueError("item ids and option ids must not overlap")
        for item in self.items:
            for gid in item.modifier_groups:
                if gid not in group_ids:
                    raise ValueError(f"item {item.id} references unknown modifier group {gid}")
        return self

    def item(self, item_id: str) -> MenuItem:
        for i in self.items:
            if i.id == item_id:
                return i
        raise KeyError(item_id)

    def group(self, group_id: str) -> ModifierGroup:
        for g in self.modifier_groups:
            if g.id == group_id:
                return g
        raise KeyError(group_id)

    def option_group(self, option_id: str) -> ModifierGroup:
        for g in self.modifier_groups:
            if any(o.id == option_id for o in g.options):
                return g
        raise KeyError(option_id)

    def option(self, option_id: str) -> Modifier:
        return self.option_group(option_id).option(option_id)

    def groups_for(self, item_id: str) -> list[ModifierGroup]:
        return [self.group(gid) for gid in self.item(item_id).modifier_groups]

    def has_item(self, item_id: str) -> bool:
        return any(i.id == item_id for i in self.items)

    def has_option(self, option_id: str) -> bool:
        return any(o.id == option_id for g in self.modifier_groups for o in g.options)

    def alias_index(self) -> list[AliasEntry]:
        """All item aliases (plus names) and option aliases, normalised, longest first."""
        entries: list[AliasEntry] = []
        for item in self.items:
            for a in [item.name, *item.aliases]:
                entries.append(AliasEntry(normalise_text(a), "item", item.id))
        for g in self.modifier_groups:
            for o in g.options:
                # Option names such as "Medium" are too generic to match on their own.
                for a in o.aliases:
                    entries.append(AliasEntry(normalise_text(a), "option", o.id))
        entries = [e for e in entries if e.alias]
        return sorted(set(entries), key=lambda e: (-len(e.alias), e.alias))

    def search(self, query: str) -> list[MenuItem]:
        q = normalise_text(query)
        if not q:
            return []
        out: list[MenuItem] = []
        for item in self.items:
            haystacks = [normalise_text(item.name), *(normalise_text(a) for a in item.aliases)]
            if any(q in h or h in q for h in haystacks):
                out.append(item)
        return out
