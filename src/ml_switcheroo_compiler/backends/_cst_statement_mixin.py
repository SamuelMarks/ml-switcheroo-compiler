"""Statement, class, comprehension, and attribute rewriting mixin for CSTTransformer."""

from __future__ import annotations

import libcst as cst

from ml_switcheroo_compiler.backends._cst_pattern_matcher import (
    _CONFIG,
    ASTPatternMatcher,
    _build_attribute_chain,
)
from ml_switcheroo_compiler.backends.transpiler_config_models import (
    FrameworkConfig,
)


class CSTStatementRewriterMixin(cst.CSTTransformer):
    """Mixin handling ClassDef, FunctionDef, comprehensions, ternary IfExp, Attribute, and AugAssign."""

    target_framework: str
    target_config: FrameworkConfig | None
    matcher: ASTPatternMatcher

    def _get_attr_chain(self, node: cst.BaseExpression) -> str:
        """Build a dot-separated string from a Name or Attribute CST node.

        Args:
            node (cst.BaseExpression): Expression node.

        Returns:
            str: Dotted attribute chain string.
        """
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            return self._get_attr_chain(node.value) + "." + node.attr.value
        return ""

    def _match_foreign_class_base(self, chain: str) -> list[str] | None:
        """Check if a base class chain matches another framework's base class and return target base parts.

        Args:
            chain (str): Dotted base class name.

        Returns:
            list[str] | None: Target base class path components if matched, else None.
        """
        for src_fw, src_config in _CONFIG.frameworks.items():
            if self.target_framework == src_fw:
                continue
            for src_k, src_v in src_config.class_bases.items():
                src_full: str = ".".join(src_v) if src_v else ""
                if chain in (src_full, src_k) and self.target_config:
                    return next(iter(self.target_config.class_bases.values()), None)
        return None

    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        """Handle Stateful-to-Functional and Functional-to-Stateful class rewrites.

        Args:
            original_node (cst.ClassDef): The original node.
            updated_node (cst.ClassDef): The updated node.

        Returns:
            cst.ClassDef: The modified class definition.
        """
        if not self.target_config:
            return updated_node

        new_bases: list[cst.Arg] = []
        mutated_bases: bool = False

        for base in updated_node.bases:
            chain: str = self._get_attr_chain(base.value)
            target_base_parts = self._match_foreign_class_base(chain)
            if target_base_parts:
                new_bases.append(base.with_changes(value=_build_attribute_chain(target_base_parts)))
                mutated_bases = True
            else:
                new_bases.append(base)

        if mutated_bases:
            return updated_node.with_changes(bases=new_bases)
        return updated_node

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        """Rewrite method names based on framework conventions.

        Args:
            original_node (cst.FunctionDef): The original node.
            updated_node (cst.FunctionDef): The updated node.

        Returns:
            cst.FunctionDef: The modified function definition.
        """
        if not self.target_config:
            return updated_node

        method_map: dict[str, str] = self.target_config.method_map
        remapped_name = self.matcher.match_function_name(updated_node.name.value, self.target_framework)
        has_self = bool(updated_node.params.params and updated_node.params.params[0].name.value == "self")
        call_dunder = "_" + "_call__"
        if remapped_name:
            if not (updated_node.name.value == call_dunder and self.target_framework == "pytorch" and not has_self):
                updated_node = updated_node.with_changes(name=cst.Name(remapped_name))
        elif updated_node.name.value in method_map:
            if updated_node.name.value == call_dunder and self.target_framework == "pytorch":
                if has_self:
                    updated_node = updated_node.with_changes(name=cst.Name(method_map[updated_node.name.value]))
            else:
                updated_node = updated_node.with_changes(name=cst.Name(method_map[updated_node.name.value]))

        # Lower early returns: if cond: return a; return b -> return where(cond, a, b)
        body_stmts = list(updated_node.body.body)
        if len(body_stmts) == 2:
            stmt1, stmt2 = body_stmts[0], body_stmts[1]
            if (
                isinstance(stmt1, cst.If)
                and not stmt1.orelse
                and len(stmt1.body.body) == 1
                and isinstance(stmt1.body.body[0], cst.SimpleStatementLine)
                and len(stmt1.body.body[0].body) == 1
                and isinstance(stmt1.body.body[0].body[0], cst.Return)
                and isinstance(stmt2, cst.SimpleStatementLine)
                and len(stmt2.body) == 1
                and isinstance(stmt2.body[0], cst.Return)
            ):
                cond = stmt1.test
                ret1 = stmt1.body.body[0].body[0].value
                ret2 = stmt2.body[0].value
                if ret1 is not None and ret2 is not None:
                    target_module_parts = self.target_config.module_path or ["jnp"]
                    where_func = _build_attribute_chain(target_module_parts + ["where"])
                    where_call = cst.Call(
                        func=where_func,
                        args=[cst.Arg(value=cond), cst.Arg(value=ret1), cst.Arg(value=ret2)],
                    )
                    new_return = cst.SimpleStatementLine(body=[cst.Return(value=where_call)])
                    updated_node = updated_node.with_changes(body=updated_node.body.with_changes(body=[new_return]))

        return updated_node

    def leave_ListComp(
        self,
        original_node: cst.ListComp,
        updated_node: cst.ListComp,
    ) -> cst.BaseExpression:
        """Functionalize list comprehensions into map operations.

        Args:
            original_node (cst.ListComp): Original list comprehension node.
            updated_node (cst.ListComp): Updated list comprehension node.

        Returns:
            cst.BaseExpression: Functionalized Map call.
        """
        for_clause = updated_node.for_in
        elt = updated_node.elt
        target = for_clause.target
        iter_expr = for_clause.iter

        if isinstance(target, cst.Name):
            lambda_func = cst.Lambda(
                params=cst.Parameters(params=[cst.Param(name=target)]),
                body=elt,
            )
            map_call = cst.Call(
                func=cst.Name("map"),
                args=[cst.Arg(value=lambda_func), cst.Arg(value=iter_expr)],
            )
            return cst.Call(
                func=cst.Name("list"),
                args=[cst.Arg(value=map_call)],
            )
        return updated_node

    def leave_GeneratorExp(
        self,
        original_node: cst.GeneratorExp,
        updated_node: cst.GeneratorExp,
    ) -> cst.BaseExpression:
        """Functionalize generator expressions into Scan/Map operations.

        Args:
            original_node (cst.GeneratorExp): Original generator expression.
            updated_node (cst.GeneratorExp): Updated generator expression.

        Returns:
            cst.BaseExpression: Functionalized generator expression.
        """
        for_clause = updated_node.for_in
        elt = updated_node.elt
        target = for_clause.target
        iter_expr = for_clause.iter

        if isinstance(target, cst.Name):
            lambda_func = cst.Lambda(
                params=cst.Parameters(params=[cst.Param(name=target)]),
                body=elt,
            )
            return cst.Call(
                func=cst.Name("map"),
                args=[cst.Arg(value=lambda_func), cst.Arg(value=iter_expr)],
            )
        return updated_node

    def leave_IfExp(
        self,
        original_node: cst.IfExp,
        updated_node: cst.IfExp,
    ) -> cst.BaseExpression:
        """Lower ternary conditional expressions into functional Cond/where calls.

        Args:
            original_node (cst.IfExp): Original ternary if node.
            updated_node (cst.IfExp): Updated ternary if node.

        Returns:
            cst.BaseExpression: Lowered functional where/Cond call.
        """
        test = updated_node.test
        body = updated_node.body
        orelse = updated_node.orelse

        target_module_parts = self.target_config.module_path if self.target_config else ["jnp"]
        where_func = _build_attribute_chain(target_module_parts + ["where"])

        return cst.Call(
            func=where_func,
            args=[
                cst.Arg(value=test),
                cst.Arg(value=body),
                cst.Arg(value=orelse),
            ],
        )

    def leave_Attribute(
        self,
        original_node: cst.Attribute,
        updated_node: cst.Attribute,
    ) -> cst.BaseExpression:
        """Stateful-to-Functional rewrites.

        Args:
            original_node (cst.Attribute): The original attribute node.
            updated_node (cst.Attribute): The updated attribute node.

        Returns:
            cst.BaseExpression: The modified expression.
        """
        matched_attr = self.matcher.match_attribute(updated_node, self.target_framework)
        if matched_attr is not None:
            return matched_attr

        if isinstance(updated_node.value, cst.Name) and updated_node.value.value == "self":
            attr_name: str = updated_node.attr.value
            return cst.Subscript(
                value=cst.Name("state"),
                slice=[cst.SubscriptElement(slice=cst.Index(value=cst.SimpleString(f'"{attr_name}"')))],
            )
        return updated_node

    def leave_AugAssign(
        self,
        original_node: cst.AugAssign,
        updated_node: cst.AugAssign,
    ) -> cst.BaseSmallStatement:
        """Functionalize in-place augmented assignments for functional target backends.

        Args:
            original_node (cst.AugAssign): The original augmented assignment node.
            updated_node (cst.AugAssign): The updated augmented assignment node.

        Returns:
            cst.BaseSmallStatement: Functionalized assignment or unchanged node.
        """
        if self.target_framework == "jax":
            op_map: dict[type[cst.BaseAugOp], cst.BaseBinaryOp] = {
                cst.AddAssign: cst.Add(),
                cst.SubtractAssign: cst.Subtract(),
                cst.MultiplyAssign: cst.Multiply(),
                cst.DivideAssign: cst.Divide(),
            }
            op_type = type(updated_node.operator)
            if op_type in op_map:
                bin_op = cst.BinaryOperation(
                    left=updated_node.target,
                    operator=op_map[op_type],
                    right=updated_node.value,
                )
                return cst.Assign(
                    targets=[cst.AssignTarget(target=updated_node.target)],
                    value=bin_op,
                )
        return updated_node
