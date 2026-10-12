# Palladium Compiler

The PAL-to-Jay compiler translates probabilistic programs (PAL) into array programs
(Jay, the JAX target language) and proves that the two semantics commute:

```
  PAL program ──────── compile ────────▶ Jay program
       │                                      │
      ⟦·⟧                                    ⟦·⟧
       │                                      │
       ▼                                      ▼
 PMF over ⟦Δ⟧    ◀──── represents ────▶  tensor of shape 𝒞⟦Δ⟧
```

A Jay tensor of shape `𝒞⟦Δ⟧` represents a PMF `μ` over the random context `Δ` when its
entry at the indices of every random environment `δ` is `μ δ`.

## Files

### Compilation (`Compile/`)

| File | Contents |
|------|----------|
| `Naming.lean` | Reserved names of compiler-generated variables (`__pal_probability_tensor`, weight and probability variables) and their distinctness. |
| `Basics.lean` | Compilation of types and operators (`Ty.compile`, `UOp.compile`, `BOp.compile`) and of the static context (`StaticContext.toJAXContext`). |
| `CompileResults.lean` | `Expr.CompileResult`: a compiled expression with its shape and a proof that the shape broadcasts to `𝒞⟦Δ⟧`. |
| `EnumSupport.lean` | `enumExpr`: the tensor of values of a random variable (`reshape(arange(n), …)`), rule C-Var. |
| `Expr.lean` | `PAL.Expr.compile` (rules C-Param, C-Const, C-Unary, C-Binary, C-Ternary, C-Var). |
| `Reduce.lean` | `reduce`, the compile-time macro `REDUCE(e, m, 0)`; shape casts. |
| `Sample.lean` | `compileSample` (rule C-Sample). |
| `Marginalize.lean` | `compileMarginalize` (rule C-Marg). |
| `Condition.lean` | `compileCondition` (rule C-Condition) and `weakenMask`, which lifts a compiled expression under a `letIn`. |
| `ReturnPerm.lean` | The permutation of a block's returned variables. |
| `Stmt.lean` | `Stmt.compile` (rules C-Skip, C-Seq, C-If, and `repeat`), `Block.compileAs`, `Block.compile` (rule C-Block). |
| `Block.lean` | The `𝒞⟦·⟧` notation for blocks. |
| `Program.lean` | `Program.compile` (rule C-Prog). |
| `EvalTrace.lean` | Shape facts about compiled marginalization. |

### Encoding of random environments as indices (`Encoding/`)

| File | Contents |
|------|----------|
| `ContextShape.lean` | `RandomContext.toShape` (`𝒞⟦Δ⟧`): the head variable of `Δ` is axis 0. |
| `EnvironmentIndices.lean` | `Environment.toIndices` (the indices of a random environment) and its inverse `Indices.toRandomEnvironment`. |
| `ToIndicesBridge.lean` | How `toIndices` interacts with inserting, permuting and renaming variables. |

### Correctness (`Correctness/`)

| File | Contents |
|------|----------|
| `WellFormed.lean` | `γ ~ θ`: a Jay valuation represents the PAL parameters. |
| `RepresentsPMF.lean` | `f ~ μ`: the denotation `f` of a tensor represents a PMF. |
| `Correct.lean` | The correctness predicates `Expr.Correct`, `Stmt.Correct`, `Block.Correct`, `Program.Correct`, stated for the denotation `⟦·⟧` of Jay. |
| `Lemmas.lean` | Shared lemmas. |
| `Expr.lean` | `Expr.correctness`. |
| `Sample.lean` | `Sample.correctness`. |
| `Marginalize.lean` | `Marginalize.correctness`. |
| `Condition.lean` | `Condition.correctness`. |
| `Stmt.lean` | `Stmt.correctness`, by induction over statements and blocks. |
| `Block.lean` | `Block.correctness`. |
| `Program.lean` | `Program.correctness`, and `Program.correctness_eval` for the evaluation of Jay in a tensor layout. |

## Axis convention

New variables are prepended to the random context: after `x ← sample …; y ← sample …`
the context is `[y, x]` and the probability tensor has shape `[|y|, |x|]`. The `i`-th
variable of the context is axis `i` of the tensor.
