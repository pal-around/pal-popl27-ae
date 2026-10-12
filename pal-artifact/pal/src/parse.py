import ast
import inspect
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Callable, cast

from .helpers import SourceInfo, source_error_context
from .syntax import (
    BinOp,
    BoolOp,
    Domain,
    Expr,
    ExprBinOp,
    ExprBoolOp,
    ExprCall,
    ExprIfThenElse,
    ExprUnaryOp,
    Literal,
    Stmt,
    StmtAssignSingle,
    StmtCondition,
    StmtIfThenElse,
    StmtMarkov,
    StmtReturn,
    StmtSample,
    StmtYield,
    UnaryOp,
    VarName,
)


@dataclass
class ParsingContext:
    # Global variables of the original function for type lookup
    global_vars: dict[str, Any]

    # Source info (filename + lines + line offset) for error reporting
    source: SourceInfo = field(default_factory=SourceInfo)

    # Parsed statements
    pal_stmts: list[Stmt] = field(default_factory=list)

    has_return: bool = field(default=False)

    non_deterministic_params: set[VarName] = field(default_factory=set)
    scalar_params: dict[VarName, Any] = field(default_factory=dict)

    # Names bound by a markov block's `xs=` to one row of a parameter per iteration.
    # They are data, not random variables, so every pass treats them like scalar
    # parameters: opaque to constant folding, invisible to liveness.
    xs_params: set[VarName] = field(default_factory=set)

    # Parsing a markov block
    in_markov_block: bool = field(default=False)
    markov_block_has_yield: bool = field(default=False)

    # Records the nested levels of markov and if-then-else statements
    with_stack: list[str] = field(default_factory=list)

    # If a markov statement's repeat argument is a scalar parameter,
    # it must be static in order to JIT compile
    static_argnames: set[VarName] = field(default_factory=set)


# Operator lookup tables for parse_expr, which runs once per AST expression node
# (millions of nodes in generated programs).
_UNARY_OP_MAP = {
    ast.Invert: UnaryOp.INVERT,
    ast.Not: UnaryOp.NOT,
    ast.USub: UnaryOp.USUB,
}
_BIN_OP_MAP = {
    ast.Add: BinOp.ADD,
    ast.Sub: BinOp.SUB,
    ast.Mult: BinOp.MUL,
    ast.Div: BinOp.DIV,
    ast.Mod: BinOp.MOD,
    ast.Pow: BinOp.POW,
}
_BOOL_OP_MAP = {ast.And: BoolOp.AND, ast.Or: BoolOp.OR}
_COMPARE_OP_MAP = {
    ast.Eq: BinOp.EQ,
    ast.NotEq: BinOp.NEQ,
    ast.Lt: BinOp.LT,
    ast.LtE: BinOp.LTE,
    ast.Gt: BinOp.GT,
    ast.GtE: BinOp.GTE,
}


def parse_dotted_name(node: ast.expr) -> str | None:
    # Resolve a chain of attribute accesses rooted at a Name into a dotted
    # string (e.g. `np.cumsum` -> "np.cumsum"). Returns None if the chain
    # bottoms out in something other than a Name (e.g. a subscript or call).
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute(value=value, attr=attr):
            parent = parse_dotted_name(value)
            if parent is None:
                return None
            return f"{parent}.{attr}"
        case _:
            return None


def _parse_probability_table(exprs: list[ast.expr], var: str, ctxt: ParsingContext) -> Expr:
    # Desugar a probability table [a, b, c, ...] (or `var > [a, b, c, ...]`) into
    # a right-nested chain of `a if var == 0 else b if var == 1 else ...`.
    if len(exprs) == 0:
        raise SyntaxError("Probability table cannot be empty")
    if len(exprs) == 1:
        return parse_expr(exprs[0], ctxt)

    ret = parse_expr(exprs[-1], ctxt)
    name = VarName(var)
    for i in reversed(range(len(exprs) - 1)):
        ret = ExprIfThenElse(ExprBinOp(name, BinOp.EQ, i), parse_expr(exprs[i], ctxt), ret)
    return ret


def parse_expr(expr: ast.expr, ctxt: ParsingContext) -> Expr:
    # Dispatch on the concrete node type by identity, most frequent types first.
    cls = type(expr)

    if cls is ast.Name:
        return VarName(cast(ast.Name, expr).id)
    if cls is ast.Constant:
        value = cast(ast.Constant, expr).value
        if not isinstance(value, Literal):
            raise SyntaxError(f"Only {Literal} literals supported")
        return value
    if cls is ast.IfExp:
        ifexp = cast(ast.IfExp, expr)
        return ExprIfThenElse(
            parse_expr(ifexp.test, ctxt),
            parse_expr(ifexp.body, ctxt),
            parse_expr(ifexp.orelse, ctxt),
        )
    if cls is ast.Compare:
        compare = cast(ast.Compare, expr)
        left = compare.left
        ops = compare.ops
        comparators = compare.comparators
        if len(ops) == 1:
            # Fast path for the overwhelmingly common single comparison.
            op0 = ops[0]
            comp0 = comparators[0]
            # Special case: `var > [a, b, c, ...]` probability-table syntax.
            if type(op0) is ast.Gt and type(left) is ast.Name and type(comp0) is ast.List:
                return _parse_probability_table(cast(ast.List, comp0).elts, cast(ast.Name, left).id, ctxt)
            return ExprBinOp(parse_expr(left, ctxt), _COMPARE_OP_MAP[op0.__class__], parse_expr(comp0, ctxt))
        operands = [parse_expr(e, ctxt) for e in [left, *comparators]]
        comparisons = [
            ExprBinOp(operands[i], _COMPARE_OP_MAP[ops[i].__class__], operands[i + 1]) for i in range(len(ops))
        ]
        return ExprBoolOp(op=BoolOp.AND, values=comparisons)
    if cls is ast.BinOp:
        binop = cast(ast.BinOp, expr)
        return ExprBinOp(
            parse_expr(binop.left, ctxt),
            _BIN_OP_MAP[binop.op.__class__],
            parse_expr(binop.right, ctxt),
        )
    if cls is ast.BoolOp:
        boolop = cast(ast.BoolOp, expr)
        return ExprBoolOp(
            op=_BOOL_OP_MAP[boolop.op.__class__],
            values=[parse_expr(v, ctxt) for v in boolop.values],
        )
    if cls is ast.UnaryOp:
        unaryop = cast(ast.UnaryOp, expr)
        return ExprUnaryOp(
            op=_UNARY_OP_MAP[unaryop.op.__class__],
            operand=parse_expr(unaryop.operand, ctxt),
        )
    if cls is ast.List:
        return _parse_probability_table(cast(ast.List, expr).elts, "_", ctxt)
    if cls is ast.Call and (name := parse_dotted_name((call := cast(ast.Call, expr)).func)) is not None:
        args = call.args
        keywords = call.keywords
        for a in chain(args, (kw.value for kw in keywords)):
            if not isinstance(a, ast.Name):
                raise SyntaxError(
                    "Generic expressions not supported in function call arguments, please assign it to a variable first"
                )
        return ExprCall(
            func=name,
            args=[VarName(cast(ast.Name, a).id) for a in args],
            kwargs={kw.arg: VarName(cast(ast.Name, kw.value).id) for kw in keywords},
        )
    raise SyntaxError("Invalid PAL expression")


def _at(parsed: Stmt, node: ast.AST) -> Stmt:
    """Give a statement synthesized while parsing ``node`` the node's source line."""
    setattr(parsed, "lineno", getattr(node, "lineno", 0))
    return parsed


def _is_init_block(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.With)
        and len(node.items) == 1
        and isinstance(node.items[0].context_expr, ast.Call)
        and isinstance(node.items[0].context_expr.func, ast.Name)
        and node.items[0].context_expr.func.id == "init"
    )


def _parse_init_block(node: ast.With, ctxt: ParsingContext) -> list[Stmt]:
    """Parse ``with init(): ...; yield x0`` -- a markov block's init block."""
    call = cast(ast.Call, node.items[0].context_expr)
    if call.args or call.keywords:
        raise SyntaxError("init() takes no arguments")
    ctxt.with_stack.append("init")
    try:
        block = [parse_stmt(s, ctxt) for s in node.body]
    finally:
        ctxt.with_stack.pop()
    if any(isinstance(s, StmtYield) for s in block[:-1]):
        raise SyntaxError("No statements allowed after yield in init block")
    if not block or not isinstance(block[-1], StmtYield):
        raise SyntaxError("Init block must end with a yield statement")
    return block


def _defined_names(stmts: list[Stmt]) -> set[VarName]:
    """Every variable bound anywhere in ``stmts``, including nested blocks."""
    names: set[VarName] = set()
    for s in stmts:
        match s:
            case StmtSample(var=var):
                names.add(var)
            case StmtAssignSingle(target_var=target):
                names.add(target)
            case StmtMarkov(vars=vs, init_block=init, statements=body):
                names |= set(vs) | _defined_names(init) | _defined_names(body)
            case StmtIfThenElse(return_vars=rvs, blocks=blocks):
                names |= set(rvs)
                for block in blocks:
                    names |= _defined_names(block)
    return names


def parse_stmt(stmt: ast.stmt, ctxt: ParsingContext) -> Stmt:
    with source_error_context(ctxt.source, getattr(stmt, "lineno", 0)):
        parsed = _parse_stmt(stmt, ctxt)
        # Stash source lineno on the parsed Stmt so the compile pass can
        # report errors against the same line.
        setattr(parsed, "lineno", getattr(stmt, "lineno", 0))
        return parsed


def _parse_stmt(stmt: ast.stmt, ctxt: ParsingContext) -> Stmt:
    if ctxt.has_return:
        raise SyntaxError("No statements allowed after return statement")
    if ctxt.in_markov_block and ctxt.markov_block_has_yield:
        raise SyntaxError("No statements allowed after yield in markov block")

    match stmt:
        # Sample (annotated-assignment syntax)
        case ast.AnnAssign(
            target=ast.Name(id=var),
            annotation=ast.Name(id=support),
            value=ast.Call(
                func=ast.Name(id="sample"),
                args=[wpp_expr],
            ),
        ):
            return StmtSample(
                VarName(var),
                Domain(support),
                parse_expr(wpp_expr, ctxt),
            )
        # Assign single (annotated-assignment syntax)
        case ast.Assign(
            targets=[ast.Name(id=target_var)],
            value=ast.Name(id=source_var),
        ):
            return StmtAssignSingle(VarName(target_var), VarName(source_var))
        # Assign (annotated-assignment syntax)
        case ast.AnnAssign(
            target=ast.Name(id=var),
            annotation=ast.Name(id=support),
            value=expr,
        ) if expr is not None:
            return StmtSample(
                VarName(var),
                Domain(support),
                ExprBinOp(VarName("_"), BinOp.EQ, parse_expr(expr, ctxt)),
                is_assign=True,
            )
        # Observe -- surface sugar for `condition`. `observe(x, v)` is exactly
        # `condition(x == v)`, and a list of values is the disjunction over them.
        # `x` is an ordinary read of the condition expression, so liveness, pruning
        # and reordering treat it like any other, and its axis is marginalized
        # when `x` dies.
        case ast.Expr(value=ast.Call(func=ast.Name(id="observe"), args=[ast.Name(id=var), value])):
            def _eq(v: Literal | VarName) -> Expr:
                return ExprBinOp(VarName(var), BinOp.EQ, v)

            match value:
                case ast.Constant(observed_value):
                    if not (isinstance(observed_value, Literal)):
                        raise SyntaxError("Observed value must be a literal, a list of literals, or a variable")
                    return StmtCondition(_eq(observed_value))
                case ast.List(elts) if all(isinstance(e, ast.Constant) for e in elts):
                    elts = cast(list[ast.Constant], elts)
                    vals = [e.value for e in elts]
                    if not all(isinstance(v, Literal) for v in vals):
                        raise SyntaxError("Observed value must be a literal, a list of literals, or a variable")
                    vals = cast(list[Literal], vals)
                    if not vals:
                        raise SyntaxError("Observed value list must not be empty")
                    if len(vals) == 1:
                        return StmtCondition(_eq(vals[0]))
                    return StmtCondition(ExprBoolOp(BoolOp.OR, [_eq(v) for v in vals]))
                case ast.Name(id=observed_var):
                    return StmtCondition(_eq(VarName(observed_var)))
                case _:
                    raise SyntaxError("Observed value must be a literal, a list of literals, or a variable")
        # Condition
        case ast.Expr(
            value=ast.Call(
                func=ast.Name(id="condition"),
                args=[expr_node],
                keywords=condition_keywords,
            )
        ):
            for kw in condition_keywords:
                if kw.arg is None:
                    raise SyntaxError("condition() does not accept **kwargs unpacking")
                raise SyntaxError(f"Unexpected keyword argument '{kw.arg}' in condition()")

            return StmtCondition(parse_expr(expr_node, ctxt))
        # Return
        case ast.Return(value):
            ctxt.has_return = True
            match value:
                case ast.Name(id=var):
                    return StmtReturn([VarName(var)])
                case ast.Tuple(elts) if all(isinstance(e, ast.Name) for e in elts):
                    elts = cast(list[ast.Name], elts)
                    vars = [VarName(e.id) for e in elts]

                    if len(set(vars)) != len(vars):
                        raise SyntaxError("Return variable names must be unique")

                    return StmtReturn(vars)
                case _:
                    raise SyntaxError("Return values must be variables")
        # Yield
        case ast.Expr(value=ast.Yield(value)):
            if ctxt.with_stack and ctxt.with_stack[-1] == "markov":
                ctxt.markov_block_has_yield = True
            match value:
                case None:
                    return StmtYield([])
                case ast.Name(id=var):
                    return StmtYield([VarName(var)])
                case ast.Tuple(elts):
                    vars = []
                    for elt in elts:
                        match elt:
                            case ast.Name(id=var):
                                vars.append(VarName(var))
                            case _:
                                raise SyntaxError("Yield values must be variables")
                    if len(set(vars)) != len(vars):
                        raise SyntaxError("Yield variable names must be unique")
                    return StmtYield(vars)
                case _:
                    raise SyntaxError("Yield values must be variables")
        # Markov
        case ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Name(id="markov"),
                        args=args,
                        keywords=keywords,
                    )
                )
            ],
            body=body_stmts,
        ):
            repeat_kws = [k for k in keywords if k.arg == "repeat"]
            xs_kws = [k for k in keywords if k.arg == "xs"]
            if len(repeat_kws) != 1 or len(xs_kws) > 1 or len(keywords) != len(repeat_kws) + len(xs_kws):
                raise SyntaxError(
                    "A markov block takes a 'repeat' keyword and an optional 'xs' keyword"
                )
            repeat = repeat_kws[0].value

            match repeat:
                case ast.Constant(value=n) if isinstance(n, int) and n >= 0:
                    repeat = n
                case ast.Name(id=var):
                    repeat = VarName(var)
                    if var in ctxt.scalar_params:
                        ctxt.static_argnames.add(VarName(var))
                case _:
                    raise SyntaxError("Markov repeat must be a non-negative integer literal or a scalar parameter")

            if ctxt.in_markov_block:
                raise SyntaxError("Nested markov blocks are not supported")

            # Per-iteration inputs: `xs=(p := theta, ...)` binds `p` to one row of the
            # parameter `theta` for each iteration -- theta's leading axis is the
            # iteration. A row is data chosen by the iteration alone (never by the loop
            # state), which is what lets the loop lower to a scan.
            xs: list[tuple[VarName, VarName]] = []
            if xs_kws:
                xs_value = xs_kws[0].value
                entries = xs_value.elts if isinstance(xs_value, ast.Tuple) else [xs_value]
                for entry in entries:
                    match entry:
                        case ast.NamedExpr(target=ast.Name(id=row), value=ast.Name(id=source)):
                            if source not in ctxt.scalar_params:
                                raise SyntaxError(
                                    f"Markov xs source '{source}' must be a parameter of the model function"
                                )
                            if row in ctxt.scalar_params or row in ctxt.xs_params:
                                raise SyntaxError(f"Markov xs name '{row}' is already in scope")
                            xs.append((VarName(row), VarName(source)))
                        case _:
                            raise SyntaxError(
                                "Markov xs entries must have the form 'row := parameter'"
                            )
                rows = [row for row, _ in xs]
                if len(set(rows)) != len(rows):
                    raise SyntaxError("Markov xs names must be unique")
                if not isinstance(repeat, int) and repeat not in ctxt.scalar_params:
                    # A random repeat count already uses the scan's xs channel for its
                    # iteration mask, and a parallel lowering needs a fixed length.
                    raise SyntaxError(
                        "A markov block with 'xs' needs a fixed repeat count: an integer "
                        "literal or a scalar parameter, not a random variable"
                    )
                ctxt.xs_params.update(rows)

            # The loop state and its init block come in two forms:
            #   with markov(x := x0, ..., repeat=n): body    -- shorthand for an init
            #                                                  block `x = x0; yield x`
            #   with markov(x, ..., repeat=n):
            #       with init(): ...; yield x0              -- the init block itself
            #       body
            walrus = [a for a in args if isinstance(a, ast.NamedExpr)]
            plain = [a for a in args if isinstance(a, ast.Name)]
            if len(walrus) + len(plain) != len(args):
                raise SyntaxError(
                    "Markov block arguments must be of the form 'var := init_var' or plain variable names"
                )
            if walrus and plain:
                raise SyntaxError(
                    "Markov block arguments must be all of the form 'var := init_var' or all plain variable names"
                )
            has_init = bool(body_stmts) and _is_init_block(body_stmts[0])

            ctxt.in_markov_block = True
            ctxt.markov_block_has_yield = False

            if walrus or not (plain or has_init):
                if has_init:
                    raise SyntaxError(
                        "A markov block whose arguments have the form 'var := init_var' cannot also "
                        "have a 'with init():' block"
                    )
                vars = []
                init_block: list[Stmt] = []
                for arg in walrus:
                    match arg:
                        case ast.NamedExpr(target=ast.Name(id=var), value=ast.Name(id=init_var)):
                            vars.append(VarName(var))
                            init_block.append(_at(StmtAssignSingle(VarName(var), VarName(init_var)), stmt))
                        case _:
                            raise SyntaxError("Markov block arguments must be of the form 'var := init_var'")
                init_block.append(_at(StmtYield(list(vars)), stmt))
                loop_body = body_stmts
            else:
                if not has_init:
                    raise SyntaxError(
                        "A markov block with plain variable arguments must begin with a 'with init():' "
                        "block that yields their initial values"
                    )
                vars = [VarName(cast(ast.Name, a).id) for a in plain]
                init_block = _parse_init_block(cast(ast.With, body_stmts[0]), ctxt)
                if len(cast(StmtYield, init_block[-1]).vars) != len(vars):
                    raise SyntaxError(
                        "Number of variables yielded by the init block must match number of markov variables"
                    )
                loop_body = body_stmts[1:]

            if len(set(vars)) != len(vars):
                raise SyntaxError("Markov variable names must be unique")

            # Recursively parse the body statements
            ctxt.with_stack.append("markov")

            parsed_body_stmts = []
            for body_stmt in loop_body:
                parsed_body_stmts.append(parse_stmt(body_stmt, ctxt))
            if not ctxt.markov_block_has_yield:
                raise SyntaxError("Markov block must contain at least one yield statement")
            if not isinstance(parsed_body_stmts[-1], StmtYield):
                raise SyntaxError("Markov block must end with a yield statement")
            if len(vars) != len(parsed_body_stmts[-1].vars):
                raise SyntaxError("Number of yielded variables must match number of markov variables")
            ctxt.in_markov_block = False
            ctxt.with_stack.pop()

            # An xs row is opaque to every later pass, so a random variable of the same
            # name would silently read the row instead.
            shadowed = _defined_names(parsed_body_stmts) & {row for row, _ in xs}
            if shadowed:
                raise SyntaxError(
                    f"variable(s) {sorted(shadowed)} shadow a markov xs name; rename them"
                )

            return StmtMarkov(
                vars=vars,
                init_block=init_block,
                repeat=repeat,
                statements=parsed_body_stmts,
                xs=xs,
            )
        # An init block outside the position where it is allowed
        case ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id="init")))]):
            raise SyntaxError("'with init():' is only allowed as the first statement of a markov block")
        # If-then-else
        case ast.With(items=withitems, body=body_stmts):
            # Parse the returned variables
            return_vars = []
            for item in withitems:
                match item:
                    case ast.withitem(
                        context_expr=ast.Name(id=var),
                    ):
                        return_vars.append(VarName(var))
                    case _:
                        raise SyntaxError("If-then-else variables must be of the form 'var1, var2, ...'")

            if len(return_vars) == 0:
                raise SyntaxError("If-then-else must return at least one variable")

            if len(set(return_vars)) != len(return_vars):
                raise SyntaxError("If-then-else variable names must be unique")

            # Parse the conditions and body statements
            conditions = []
            blocks = []
            yields = []

            def parse_body_stmts(body_stmts: list[ast.stmt]):
                def helper(body_stmts: list[ast.stmt]):
                    # Parse body statements
                    ctxt.with_stack.append("if")
                    block = [parse_stmt(s, ctxt) for s in body_stmts]
                    ctxt.with_stack.pop()
                    blocks.append(block)

                    # Parse yield statement (should be last statement in block)
                    has_yield = False
                    for stmt in block:
                        if has_yield:
                            raise SyntaxError("No statements allowed after yield in if-then-else block")

                        match stmt:
                            case StmtYield(vars):
                                has_yield = True
                                yields.append(vars)

                    if not has_yield:
                        raise SyntaxError("If-then-else block must end with a yield statement")

                match body_stmts:
                    case [ast.If(test, body, orelse)]:
                        # Add condition
                        conditions.append(parse_expr(test, ctxt))

                        helper(body)

                        # Recursively parse orelse
                        parse_body_stmts(orelse)
                    case _:  # Else block
                        helper(body_stmts)

            parse_body_stmts(body_stmts)

            # Sanity checks
            assert len(conditions) + 1 == len(blocks)
            assert len(blocks) == len(yields)

            # Check that all yield statements yield the same number of variables as return_vars
            if any(len(yield_vars) != len(return_vars) for yield_vars in yields):
                raise SyntaxError(
                    "Number of yielded variables in if-then-else blocks must match number of return variables"
                )

            return StmtIfThenElse(
                return_vars=return_vars,
                conditions=conditions,
                blocks=blocks,
                yields=yields,
            )
        case _:
            raise SyntaxError("Invalid PAL statement")


def parse_pal(tree: ast.stmt, ctxt: ParsingContext):
    tree_lineno = getattr(tree, "lineno", 1)
    with source_error_context(ctxt.source, tree_lineno):
        match tree:
            case ast.FunctionDef() as func:
                for stmt in func.body:
                    ctxt.pal_stmts.append(parse_stmt(stmt, ctxt))
            case _:
                raise SyntaxError("PAL can only be applied to function definitions")
        if not ctxt.has_return:
            raise SyntaxError("PAL functions must have a return statement")


def parse_pal_parameters(f: Callable, ctxt: ParsingContext):
    for t in f.__type_params__:
        ctxt.non_deterministic_params.add(VarName(t.__name__))

    for param in inspect.signature(f).parameters.values():
        ctxt.scalar_params[VarName(param.name)] = None if param.default is param.empty else param.default
