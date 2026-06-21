"""Aliases for misc."""


class _C_Class:
    def __getitem__(self, key: object) -> object:
        raise NotImplementedError("c_ is not fully supported yet.")


class _R_Class:
    def __getitem__(self, key: object) -> object:
        raise NotImplementedError("r_ is not fully supported yet.")


class _S_Class:
    def __getitem__(self, key: object) -> object:
        return key


class _IndexExp_Class:
    def __getitem__(self, key: object) -> object:
        return key


flexible = "flexible"


class _MgridClass_mgrid:
    def __getitem__(self, key: object) -> object:
        raise NotImplementedError("mgrid not fully supported")


class _MgridClass_ogrid:
    def __getitem__(self, key: object) -> object:
        raise NotImplementedError("ogrid not fully supported")
