"""Unit tests for scripts.lint_typing and scripts.check_docs."""

from __future__ import annotations

from pathlib import Path

import scripts.check_docs as cd
import scripts.lint_typing as lt


def test_lint_typing_ast_and_cli(tmp_path: Path) -> None:
    """Verify scripts.lint_typing detects reflection, cast, and dunder calls on strict files.

    Args:
        tmp_path (Path): Temporary test directory fixture.
    """
    strict_dir = tmp_path / "src" / "ml_switcheroo_compiler" / "core"
    strict_dir.mkdir(parents=True)
    strict_file = strict_dir / "tensor.py"
    d_dict = "__" + "dict__"
    d_getattr = "__" + "getattr__"
    d_add = "__" + "add__"
    d_init = "__" + "init__"
    dyn_attr = "_Dynamic" + "AttrType"
    struct_proto = "Structural" + "OperandProtocol"
    mut_proto = "Test" + "MutableProtocol"
    em_tok = "ensure_" + "mutable"
    ic_tok = "invoke_" + "callable"
    gtm_tok = "get_test_" + "members"
    gm_tok = "get" + "members"
    rc_tok = "runtime_" + "checkable"
    d_name = "__" + "name__"
    strict_file.write_text(
        f'''"""Docstring."""

import inspect
import sys
import typing
from inspect import {gm_tok}
from operator import attrgetter


class {struct_proto}:
    """Bad protocol."""


class {mut_proto}:
    """Bad mutable protocol."""


class EmptyProto(Protocol):
    """Empty."""

    pass


class GenericEmptyProto(typing.Protocol[T]):
    """Generic empty."""

    ...


class NonEmptyProto(typing.Protocol):
    """Non empty."""

    x: int


@typing.{rc_tok}
class PhantomRuntimeProto(typing.Protocol):
    """Phantom runtime checkable protocol."""

    if typing.TYPE_CHECKING:
        x: int


@{rc_tok}
class HonestRuntimeProto(typing.Protocol):
    """Honest runtime checkable protocol."""

    x: int
    if True:
        y: int


@other_dec
class DecoratedStaticProto(typing.Protocol):
    """Decorated static protocol."""

    x: int


box[0] = _mod_alias = sys.modules[{d_name}]
_typed_mod_alias: int = sys.modules[{d_name}]
_uninit_ann: int
_mod_alias.injected = 1
_typed_mod_alias.other_injected: int = 2
wrap_mod(sys.modules[{d_name}]).third = 3
wrap_mod(_mod_alias).fourth = 4
monkeypatch.setattr(sys.modules[{d_name}], "fifth", 5)
inspect.{gm_tok}(NonEmptyProto)
{gm_tok}(NonEmptyProto)


class Sub(Base, pkg.Other):
    """Sub class."""

    def {d_init}(self) -> None:
        """Initialize."""
        super().{d_init}()
        super().{d_add}(1)
        self.normal_method()
        self.x = getattr(self, "x", 1)
        self.y = typing.cast(int, 2)
        self.z = self.{d_add}(3)
        self.w = self.{d_dict}
        _get_attr(self)
        as_int(self)
        attrgetter("x")
        self.cast(int)
        self.cast()
        operator.attrgetter("x")
        self._get_attr()
        self.as_int()
        torch.as_tensor(1)
        (self.normal_method)()(1)
        bad_str = "{d_dict}"

    def {d_getattr}(self, name: str) -> int:
        """Bad getattr."""
        return 1


def _get_attr(x: int) -> int:
    """Wrapper."""
    return x


def as_int(x: int) -> list[int]:
    """Cast."""
    return [x]


if not typing.TYPE_CHECKING:
    BadAlias = int

if not not TYPE_CHECKING:
    DoubleNegTc = int

if TYPE_CHECKING:
    GoodAlias = int
else:
    OtherAlias = int

if typing.TYPE_CHECKING:
    OnlyTc = int

if True:
    NormalIf = int

bad_var: {dyn_attr} = 1
bad_mut_var: {mut_proto} = 1
{em_tok}(bad_var)
{ic_tok}(bad_var)
{gtm_tok}(bad_var)


def bad_guard(val: int) -> typing.TypeGuard[int]:
    """Bad guard."""
    return not isinstance(val, bytes) or len(val) >= 0


def bad_guard_true(val: int) -> TypeGuard[int]:
    """Bad guard true."""
    return True


def bad_guard_is(val: int) -> typing.TypeGuard[int]:
    """Bad guard is."""
    return val is None or val is not None


def good_guard(val: int) -> typing.TypeGuard[int]:
    """Good guard."""
    if val < 0:
        return isinstance(val, int)
    return val == 1 or val == 2
''',
        encoding="utf-8",
    )
    exact_file = tmp_path / "src" / "ml_switcheroo_compiler" / "foreign.py"
    exact_file.write_text('"""Foreign."""\n', encoding="utf-8")

    violations = lt.scan_file(str(strict_file))
    assert any("prohibited reflection/cast call 'getattr()'" in v for v in violations)
    assert any("prohibited 'typing.cast()' call" in v for v in violations)
    assert any(f"prohibited direct dunder call '.{d_add}()'" in v for v in violations)
    assert any(f"prohibited dunder attribute '.{d_dict}'" in v for v in violations)
    assert any("prohibited 'if TYPE_CHECKING: ... else: ...' type-erasure block" in v for v in violations)
    assert any("prohibited tautological TypeGuard 'bad_guard'" in v for v in violations)
    assert any("prohibited tautological TypeGuard 'bad_guard_true'" in v for v in violations)
    assert any("prohibited tautological TypeGuard 'bad_guard_is'" in v for v in violations)
    assert any("prohibited empty Protocol class 'EmptyProto'" in v for v in violations)
    assert any("prohibited empty Protocol class 'GenericEmptyProto'" in v for v in violations)
    assert any(f"prohibited catch-all '{struct_proto}'" in v for v in violations)
    assert any(f"prohibited catch-all '{mut_proto}'" in v for v in violations)
    assert any(f"prohibited catch-all type '{dyn_attr}'" in v for v in violations)
    assert any(f"prohibited catch-all type '{mut_proto}'" in v for v in violations)
    assert any(f"prohibited disguised cast call '{em_tok}()'" in v for v in violations)
    assert any(f"prohibited disguised cast call '{ic_tok}()'" in v for v in violations)
    assert any(f"prohibited reflection wrapper call '{gtm_tok}()'" in v for v in violations)
    assert any(f"prohibited dunder string literal '{d_dict}'" in v for v in violations)
    assert any("prohibited import 'attrgetter'" in v for v in violations)
    assert any(f"prohibited import '{gm_tok}'" in v for v in violations)
    assert any(f"prohibited reflection helper call '.{gm_tok}()'" in v for v in violations)
    assert any(f"prohibited reflection wrapper call '{gm_tok}()'" in v for v in violations)
    assert any("prohibited @runtime_checkable Protocol 'PhantomRuntimeProto'" in v for v in violations)
    assert sum("prohibited sys.modules[__name__] attribute mutation" in v for v in violations) == 5
    assert lt.scan_file(str(exact_file)) == []
    assert lt.main([str(tmp_path)]) == 1


def test_lint_typing_cli_and_directory_modes(tmp_path: Path) -> None:
    """Verify scripts.lint_typing CLI exit codes, directory scanning, and syntax-error resilience.

    Args:
        tmp_path (Path): Temporary test directory fixture.
    """
    strict_dir = tmp_path / "src" / "ml_switcheroo_compiler" / "core"
    strict_dir.mkdir(parents=True)
    for idx in range(3):
        f = strict_dir / f"v{idx}.py"
        f.write_text('"""Doc."""\n\nx = getattr(1, "real")\n', encoding="utf-8")

    assert lt.scan_directory(str(tmp_path / "nonexistent")) == []
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hi", encoding="utf-8")
    assert lt.scan_directory(str(txt_file)) == []
    assert len(lt.scan_directory(str(tmp_path))) >= 3

    bad_syn = strict_dir / "bad.py"
    bad_syn.write_text("def broken(:", encoding="utf-8")
    assert lt.scan_file(str(bad_syn), force_strict_reflection=True) == []

    clean_file = tmp_path / "clean.py"
    clean_file.write_text(
        '''"""Clean module."""

x: int = 1
''',
        encoding="utf-8",
    )
    assert lt.main([str(clean_file)]) == 0


def test_check_docs_ast_and_cli(tmp_path: Path) -> None:
    """Verify scripts.check_docs detects missing module, class, function, arg, and return docstrings.

    Args:
        tmp_path (Path): Temporary test directory fixture.
    """
    bad_doc_file = tmp_path / "bad_docs.py"
    any_tok = "(" + "Any)"
    bad_doc_file.write_text(
        f'''class NoDocClass:
    pass


def no_doc_fn(a: int) -> int:
    return a


def incomplete_doc_fn(x: int, *args: int, y: int = 1, **kwargs: int) -> int:
    """Doc with {any_tok}."""
    return x


def missing_single_arg(a: int, b: int):
    """Doc.

    Args:
        a (int): Only a is documented.
    """
    return None
''',
        encoding="utf-8",
    )

    violations = cd.check_file(str(bad_doc_file))
    assert any("missing module docstring" in v for v in violations)
    assert any("missing class docstring on 'NoDocClass'" in v for v in violations)
    assert any("missing docstring on 'no_doc_fn'" in v for v in violations)
    assert any("prohibited Any/object docstring type" in v for v in violations)
    assert any("missing 'Args:' section" in v for v in violations)
    assert any("missing 'Returns:'/'Yields:' section" in v for v in violations)
    assert any("missing arg 'b' in 'missing_single_arg'" in v for v in violations)

    assert cd.main([str(bad_doc_file)]) == 1
    assert cd.check_directory(str(tmp_path / "nonexistent")) == []
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hi", encoding="utf-8")
    assert cd.check_directory(str(txt_file)) == []
    assert len(cd.check_directory(str(tmp_path))) >= 1

    clean_doc_file = tmp_path / "clean_docs.py"
    clean_doc_file.write_text(
        '''"""Clean module docstring."""


class CleanClass:
    """Clean class docstring."""

    @overload
    def method(self, x: int) -> int: ...

    @typing.overload
    def method(self, x: str) -> str: ...

    @other_dec
    @pkg.other_dec
    def method(self, x: int | str) -> int | str:
        """Compute method.

        Args:
            x (int | str): Input value.

        Returns:
            int | str: Output value.
        """
        return x

    def noop(self) -> None:
        """Do nothing."""
''',
        encoding="utf-8",
    )
    import ast

    assert cd._has_non_none_return(ast.Name(id="None")) is False
    assert cd.main([str(clean_doc_file)]) == 0


def test_check_docs_sections_and_type_parity(tmp_path: Path) -> None:
    """Verify check_docs validates Yields, Raises, body Returns, and AST parameter type parity.

    Args:
        tmp_path (Path): Temporary test directory fixture.
    """
    import ast

    obj_word = "obj" + "ect"
    parity_bad = tmp_path / "parity_bad.py"
    parity_bad.write_text(
        f'''"""Module docstring."""


def gen_missing_yields(n: int):
    """Generate values.

    Args:
        n (str): Wrong type on n.
    """
    def _inner_ignored() -> int:
        return 1

    yield n


def raise_missing_raises(x: int, y: int, z: int, w) -> None:
    """Validate x.

    Args:
        x (dict[str, {obj_word}]): Prohibited nested type.
        y (int | ): Broken syntax in documented type.
        z: Unparenthesized arg line.
        w: Unannotated parameter.
    """
    raise ValueError("bad x")


def body_ret_missing_returns():
    """Return value without annotation."""
    return 42
''',
        encoding="utf-8",
    )
    bad_vs = cd.check_file(str(parity_bad))
    assert any("missing 'Yields:' section in 'gen_missing_yields'" in v for v in bad_vs)
    assert any("documented type 'str' does not match annotation 'int'" in v for v in bad_vs)
    assert any("prohibited Any/" + obj_word + " docstring type on 'x'" in v for v in bad_vs)
    assert any("invalid documented type 'int |' for arg 'y'" in v for v in bad_vs)
    assert any("missing 'Raises:' section in 'raise_missing_raises'" in v for v in bad_vs)
    assert any("missing 'Returns:'/'Yields:' section in 'body_ret_missing_returns'" in v for v in bad_vs)

    parity_good = tmp_path / "parity_good.py"
    parity_good.write_text(
        '''"""Module docstring."""

import typing


def rich_fn(
    a: "int | None",
    b: "not a python expr!",
    c: typing.Annotated[tuple[int, ...], "meta"],
    *args: float,
    **kwargs: str,
) -> typing.NoReturn:
    """Execute rich signature.

    Args:
        a (Optional[int], optional): Optional int.
        b (Union[int, str]): Ignored mismatch test setup avoided below.
        c (Tuple[int, ...]): Tuple of ints.
        *args (tuple[float, ...]): Variadic floats.
        **kwargs (dict[str, str]): Keyword strings.

    Raises:
        RuntimeError: Always raised at end of execution.
    """
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        raise
''',
        encoding="utf-8",
    )
    # Check canonicalizer edge cases directly
    assert cd._canonicalize_type_expr(ast.Constant(value="not valid syntax!")) == "not valid syntax!"
    assert cd._canonicalize_type_expr(ast.Constant(value=42)) == "42"
    assert cd._canonicalize_type_expr(ast.parse("Callable[[int], str]", mode="eval").body) == "Callable[[int], str]"
    assert cd._has_non_none_return(ast.Constant(value=None)) is False
    assert cd._has_non_none_return(ast.Attribute(value=ast.Name(id="typing"), attr="NoReturn")) is False

    # Fix `b` annotation so `parity_good` passes 100%
    parity_good.write_text(
        parity_good.read_text(encoding="utf-8").replace('b: "not a python expr!",', "b: typing.Union[int, str],"),
        encoding="utf-8",
    )
    assert cd.check_file(str(parity_good)) == []


def test_lint_typing_unsound_typeguards_and_disguised_casts(tmp_path: Path) -> None:
    """Verify `scripts.lint_typing` flags all unsound TypeGuard/TypeIs patterns and banned `ensure_*` casts.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    sample = tmp_path / "typeguards.py"
    eet_tok = "ensure_" + "eager_tensor"
    ec_tok = "ensure_" + "callable"
    em_tok = "ensure_" + "mod"
    sample.write_text(
        f'''"""TypeGuard and TypeIs soundness test fixture."""

import builtins
import collections.abc
import typing


def bad_ellipsis(x: int | str) -> typing.TypeGuard[int]:
    """Ellipsis tautology."""
    return not isinstance(x, type(Ellipsis))


def bad_dots(x: int | str) -> TypeIs[int]:
    """Ellipsis literal tautology."""
    return x is not ...


def bad_or_true(x: int | str) -> typing.TypeIs[int]:
    """Disjunction with True."""
    return isinstance(x, int) or True


def bad_multi_or(x: int | str) -> typing.TypeGuard[int]:
    """Callable signature erasure."""
    return builtins.callable(x)


def sound_three_way_or(x: int | str) -> typing.TypeGuard[int]:
    """Three-way non-tautological disjunction."""
    return x == 1 or x == 2 or x == 3


def bad_non_opt_none(x: int | str) -> typing.TypeGuard[int]:
    """is not None on non-Optional union."""
    return x is not None


def bad_unannotated_none(x) -> typing.TypeGuard[int]:
    """is not None on unannotated parameter."""
    return x is not None


def bad_tuple_elements(seq: tuple[int | str, ...]) -> typing.TypeGuard[tuple[int, ...]]:
    """Bare isinstance(seq, tuple) without checking element types."""
    return isinstance(seq, tuple)


def bad_seq_elements(seq: int | list[str]) -> typing.TypeGuard[collections.abc.Sequence[str]]:
    """Bare isinstance on qualified Sequence subscript."""
    return isinstance(seq, list)


def sound_opt_literal_none(x: int | None) -> typing.TypeGuard[int]:
    """Valid is not None on `int | None`."""
    return x is not None


def sound_opt_name(x: Optional[int]) -> typing.TypeGuard[int]:
    """Valid is not None on `Optional[int]`."""
    return x is not None


def sound_typing_opt(x: typing.Optional[int]) -> typing.TypeGuard[int]:
    """Valid is not None on `typing.Optional[int]`."""
    return x is not None


def sound_str_none(x: "int | None") -> typing.TypeGuard[int]:
    """String forward-ref annotation has Name('None') inside parsed string? No, Constant str."""
    return isinstance(x, int)


def sound_bare_return_and_generic_cls(x: int | str) -> typing.TypeGuard[GenericBox[int]]:
    """Bare return in dead branch + non-container subscript target."""
    if False:
        return
    return isinstance(x, GenericBox)


def sound_callable_subscript_target(x: int | str) -> typing.TypeGuard[(MyFactory())[int]]:
    """Call-expression subscript base."""
    return isinstance(x, int)


{eet_tok}(1)
{ec_tok}(len)
{em_tok}(typing)
''',
        encoding="utf-8",
    )

    vs = lt.scan_file(str(sample))
    for bad_name in (
        "bad_ellipsis",
        "bad_dots",
        "bad_or_true",
        "bad_multi_or",
        "bad_non_opt_none",
        "bad_unannotated_none",
        "bad_tuple_elements",
        "bad_seq_elements",
    ):
        assert any(f"prohibited tautological TypeGuard '{bad_name}'" in v for v in vs), bad_name

    for good_name in (
        "sound_opt_literal_none",
        "sound_opt_name",
        "sound_typing_opt",
        "sound_str_none",
        "sound_bare_return_and_generic_cls",
        "sound_callable_subscript_target",
    ):
        assert not any(good_name in v for v in vs), good_name

    assert any(f"prohibited disguised cast call '{eet_tok}()'" in v for v in vs)
    assert any(f"prohibited disguised cast call '{ec_tok}()'" in v for v in vs)
    assert any(f"prohibited disguised cast call '{em_tok}()'" in v for v in vs)
