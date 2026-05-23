from dataclasses import dataclass, field
from functools import lru_cache

from shaderbox.constants import RESOURCES_DIR

_EMOJI_TEST_FILE = RESOURCES_DIR / "emoji" / "emoji-test.txt"
# "Component" holds skin-tone modifiers etc. — not standalone picker entries.
_SKIP_GROUPS = frozenset({"Component"})


@dataclass
class EmojiEntry:
    char: str
    name: str


@dataclass
class EmojiGroup:
    name: str
    entries: list[EmojiEntry] = field(default_factory=list)


@lru_cache(maxsize=1)
def load_emoji_groups() -> list[EmojiGroup]:
    """Parse the vendored Unicode `emoji-test.txt` into ordered groups.

    Keeps only fully-qualified rows (the native-picker set); each entry's `char`
    is the WHOLE codepoint sequence (ZWJ/skin-tone/flag intact), never split.
    """
    groups: list[EmojiGroup] = []
    current: EmojiGroup | None = None

    with _EMOJI_TEST_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# group:"):
                name: str = line.split(":", 1)[1].strip()
                current = EmojiGroup(name=name)
                if name not in _SKIP_GROUPS:
                    groups.append(current)
                continue
            if not line or line.startswith("#"):
                continue
            if "; fully-qualified" not in line:
                continue
            if current is None or current.name in _SKIP_GROUPS:
                continue
            # Format: <codepoints> ; fully-qualified  # <emoji> E<ver> <name>
            comment: str = line.split("#", 1)[1].strip()
            parts: list[str] = comment.split(" ", 2)
            if len(parts) < 3:
                continue
            char: str = parts[0]
            entry_name: str = parts[2]
            current.entries.append(EmojiEntry(char=char, name=entry_name))

    return groups
