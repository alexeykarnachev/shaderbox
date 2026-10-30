"""The exported function names of a shared library, read from the file itself.

Both formats are parsed here, so either host reads either binary: a Linux box checks the
vendored `.dll` without a Windows toolchain, and a Windows box needs no `nm` for the `.so`.
"""

import struct
from pathlib import Path

_SHT_DYNSYM = 11
_STT_FUNC = 2
_STB_GLOBAL = 1
_SHN_UNDEF = 0
_PE32_PLUS = 0x20B


def _cstr(data: bytes, offset: int) -> str:
    return data[offset : data.index(b"\0", offset)].decode()


def _elf_exports(data: bytes) -> set[str]:
    assert data[4] == 2, "only 64-bit ELF is vendored"
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
        for i in range(shnum)
    ]
    names: set[str] = set()
    for _, sh_type, _, _, offset, size, link, _, _, entsize in sections:
        if sh_type != _SHT_DYNSYM:
            continue
        strtab = sections[link][4]
        for at in range(offset, offset + size, entsize):
            st_name, st_info, _, st_shndx = struct.unpack_from("<IBBH", data, at)
            if (
                st_shndx != _SHN_UNDEF
                and st_info >> 4 == _STB_GLOBAL
                and st_info & 0xF == _STT_FUNC
            ):
                names.add(_cstr(data, strtab + st_name))
    return names


def _pe_exports(data: bytes) -> set[str]:
    (pe,) = struct.unpack_from("<I", data, 0x3C)
    assert data[pe : pe + 4] == b"PE\0\0"
    nsections, _, _, _, optsize, _ = struct.unpack_from("<HIIIHH", data, pe + 6)
    optional = pe + 24
    (magic,) = struct.unpack_from("<H", data, optional)
    assert magic == _PE32_PLUS, "only 64-bit PE is vendored"
    export_rva, _ = struct.unpack_from("<II", data, optional + 112)
    table = optional + optsize
    spans = [
        struct.unpack_from("<IIII", data, table + i * 40 + 8) for i in range(nsections)
    ]

    def offset(rva: int) -> int:
        for vsize, vaddr, rawsize, rawptr in spans:
            if vaddr <= rva < vaddr + max(vsize, rawsize):
                return rva - vaddr + rawptr
        raise ValueError(f"RVA {rva:#x} lies in no section")

    if export_rva == 0:
        return set()
    directory = offset(export_rva)
    (count,) = struct.unpack_from("<I", data, directory + 24)
    (names_rva,) = struct.unpack_from("<I", data, directory + 32)
    names_at = offset(names_rva)
    return {
        _cstr(data, offset(struct.unpack_from("<I", data, names_at + 4 * i)[0]))
        for i in range(count)
    }


def exported_functions(path: Path) -> set[str]:
    data = path.read_bytes()
    if data[:4] == b"\x7fELF":
        return _elf_exports(data)
    if data[:2] == b"MZ":
        return _pe_exports(data)
    raise ValueError(f"{path.name} is neither ELF nor PE")
