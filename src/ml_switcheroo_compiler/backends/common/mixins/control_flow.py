"""Module control_flow.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Provide mixin module."""
from typing import Any

from .common import CommonASTVisitor


class ControlFlowASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Control flow AST generator mixin."""

    def visit_Scan(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_Scan operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        # Natively, backends implement this as a specific scan.
        return f"{pfx}_scan({', '.join(input_vars)})"

    def visit_Switch(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_Switch operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        # Fallback to a custom runner
        return f"{pfx}_switch({', '.join(input_vars)})"

    def visit_TimeDistributed(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_TimeDistributed operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        # Fallback implementation: we assume the frontend has provided a TimeDistributed node.
        # Natively, backends might want to generate a loop or a vmap.
        # For simplicity in this mixin, we return a function call to a backend-specific time_distributed utility.
        return f"{self.generator.get_fallback_prefix()}_time_distributed({input_vars[0]}, '{node.attributes.get('wrapped_op_name', '')}')"

    def visit_Assert(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_Assert operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        data = kwargs.get("data", ["Assertion failed."])
        return f"{pfx}_assert({input_vars[0]}, data={data})"

    def visit_AssociativeScan(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AssociativeScan operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        return f"{pfx}_associative_scan({', '.join(input_vars)})"

    def visit_WhileLoop(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """visit_WhileLoop function.

        Args:
            node: The node.
            input_vars: The input variables.
            kwargs: Additional kwargs.

        Args:
            message (str): The message.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.
        node (Any): The node parameter.
        input_vars (Any): The input_vars parameter.

        Returns:
        Any: Result.
        """
        from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor

        cond_graph = node.attributes.get("cond")
        body_graph = node.attributes.get("body")
        state_var = input_vars[0]

        # Cond Fn
        self.generator.add_line(f"def cond_fn_{node.id}(state):")
        self.generator.indent_level += 1
        cond_gen = self.generator.__class__(cond_graph)
        cond_gen.formatter.indent_level = self.generator.indent_level
        visitor = CodeGeneratorVisitor(cond_gen)
        visitor.generate_body("state")
        for line in cond_gen.code[1:]:
            self.generator.add_line(line)
        self.generator.indent_level -= 1

        # Body Fn
        self.generator.add_line(f"def body_fn_{node.id}(state):")
        self.generator.indent_level += 1
        body_gen = self.generator.__class__(body_graph)
        body_gen.formatter.indent_level = self.generator.indent_level
        visitor = CodeGeneratorVisitor(body_gen)
        visitor.generate_body("state")
        for line in body_gen.code[1:]:
            self.generator.add_line(line)
        self.generator.indent_level -= 1

        self.generator.add_line(f"loop_val_{node.id} = {state_var}")
        self.generator.add_line(f"while cond_fn_{node.id}(loop_val_{node.id}):")
        self.generator.indent_level += 1
        self.generator.add_line(f"loop_val_{node.id} = body_fn_{node.id}(loop_val_{node.id})")
        self.generator.indent_level -= 1

        return f"loop_val_{node.id}"

    def visit_Cond(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """visit_Cond function.

        Args:
            node: The node.
            input_vars: The input variables.
            kwargs: Additional kwargs.

        Args:
            message (str): The message.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.
        node (Any): The node parameter.
        input_vars (Any): The input_vars parameter.

        Returns:
        Any: Result.
        """
        from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor

        cond_val = input_vars[0]
        true_graph = node.attributes.get("true_branch") or node.attributes.get("then_branch")
        false_graph = node.attributes.get("false_branch") or node.attributes.get("else_branch")

        # True Branch
        self.generator.add_line(f"def true_fn_{node.id}():")
        self.generator.indent_level += 1
        true_gen = self.generator.__class__(true_graph)
        true_gen.formatter.indent_level = self.generator.indent_level
        visitor = CodeGeneratorVisitor(true_gen)
        visitor.generate_body()
        for line in true_gen.code[1:]:
            self.generator.add_line(line)
        self.generator.indent_level -= 1

        # False Branch
        self.generator.add_line(f"def false_fn_{node.id}():")
        self.generator.indent_level += 1
        false_gen = self.generator.__class__(false_graph)
        false_gen.formatter.indent_level = self.generator.indent_level
        visitor = CodeGeneratorVisitor(false_gen)
        visitor.generate_body()
        for line in false_gen.code[1:]:
            self.generator.add_line(line)
        self.generator.indent_level -= 1

        return f"true_fn_{node.id}() if {cond_val} else false_fn_{node.id}()"

    def visit_ForiLoop(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """visit_ForiLoop function.

        Args:
            node: The node.
            input_vars: The input variables.
            kwargs: Additional kwargs.

        Args:
            message (str): The message.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.
        node (Any): The node parameter.
        input_vars (Any): The input_vars parameter.

        Returns:
        Any: Result.
        """
        from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor

        lower_bound = input_vars[0]
        upper_bound = input_vars[1]
        init_val = input_vars[2]
        body_graph = node.attributes.get("body")

        # Body Fn
        self.generator.add_line(f"def body_fn_{node.id}(i, state):")
        self.generator.indent_level += 1
        body_gen = self.generator.__class__(body_graph)
        body_gen.formatter.indent_level = self.generator.indent_level
        visitor = CodeGeneratorVisitor(body_gen)
        # Assuming args passed as list [i, state]
        visitor.generate_body("[i, state]")
        for line in body_gen.code[1:]:
            self.generator.add_line(line)
        self.generator.indent_level -= 1

        self.generator.add_line(f"loop_val_{node.id} = {init_val}")
        self.generator.add_line(f"for i_{node.id} in range({lower_bound}, {upper_bound}):")
        self.generator.indent_level += 1
        self.generator.add_line(f"loop_val_{node.id} = body_fn_{node.id}(i_{node.id}, loop_val_{node.id})")
        self.generator.indent_level -= 1

        return f"loop_val_{node.id}"

    def visit_Map(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """visit_Map function.

        Args:
            node: The node.
            input_vars: The input variables.
            kwargs: Additional kwargs.

        Args:
            message (str): The message.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.
        node (Any): The node parameter.
        input_vars (Any): The input_vars parameter.

        Returns:
        Any: Result.
        """
        from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor

        xs = input_vars[0]
        f_graph = node.attributes.get("f")

        self.generator.add_line(f"def map_fn_{node.id}(x):")
        self.generator.indent_level += 1
        f_gen = self.generator.__class__(f_graph)
        f_gen.formatter.indent_level = self.generator.indent_level
        visitor = CodeGeneratorVisitor(f_gen)
        visitor.generate_body("[x]")
        for line in f_gen.code[1:]:
            self.generator.add_line(line)
        self.generator.indent_level -= 1

        pfx = self.generator.get_fallback_prefix()
        # map usually means array mapping over first dimension
        return f"{pfx}.stack([map_fn_{node.id}(x) for x in {xs}])"

    def visit_Fold(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """visit_Fold function.

        Args:
            node: The node.
            input_vars: The input variables.
            kwargs: Additional kwargs.

        Args:
            message (str): The message.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.
        node (Any): The node parameter.
        input_vars (Any): The input_vars parameter.

        Returns:
        Any: Result.
        """
        from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor

        init = input_vars[0]
        xs = input_vars[1]
        f_graph = node.attributes.get("f")

        self.generator.add_line(f"def fold_fn_{node.id}(acc, x):")
        self.generator.indent_level += 1
        f_gen = self.generator.__class__(f_graph)
        f_gen.formatter.indent_level = self.generator.indent_level
        visitor = CodeGeneratorVisitor(f_gen)
        visitor.generate_body("[acc, x]")
        for line in f_gen.code[1:]:
            self.generator.add_line(line)
        self.generator.indent_level -= 1

        self.generator.add_line(f"fold_val_{node.id} = {init}")
        self.generator.add_line(f"for x_{node.id} in {xs}:")
        self.generator.indent_level += 1
        self.generator.add_line(f"fold_val_{node.id} = fold_fn_{node.id}(fold_val_{node.id}, x_{node.id})")
        self.generator.indent_level -= 1

        return f"fold_val_{node.id}"

    def visit_Vmap(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """visit_Vmap function.

        Args:
            node: The node.
            input_vars: The input variables.
            kwargs: Additional kwargs.

        Args:
            message (str): The message.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.
        node (Any): The node parameter.
        input_vars (Any): The input_vars parameter.

        Returns:
        Any: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        return f"{pfx}_vmap({', '.join(input_vars)})"

    def visit_Pmap(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """visit_Pmap function.

        Args:
            node: The node.
            input_vars: The input variables.
            kwargs: Additional kwargs.

        Args:
            message (str): The message.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.
        node (Any): The node parameter.
        input_vars (Any): The input_vars parameter.

        Returns:
        Any: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        return f"{pfx}_pmap({', '.join(input_vars)})"
