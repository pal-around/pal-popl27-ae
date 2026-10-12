# Palladium: a Verified Compiler from PAL to Jay

This artifact is the Lean 4 mechanization accompanying the paper. It contains the
syntax, type systems and denotational semantics of the probabilistic language PAL and
of the array language Jay (modelled on JAX), the compiler from PAL to Jay, and the
machine-checked proofs that compilation preserves semantics.

Copyright 2026 the authors. Licensed under the MIT License (see [`LICENSE`](LICENSE)).

This README follows the POPL 2027 artifact-evaluation guidelines:

1. [Claims](#1-claims): the paper's mechanized definitions and theorems and where they
   are in the code.
2. [Getting started](#2-getting-started): installation and a quick sanity test.
3. [Evaluation instructions](#3-evaluation-instructions): re-checking the proofs,
   replaying them in the Lean kernel, and auditing their assumptions.
4. [Additional description](#4-additional-description): the structure of the
   development, its correspondence with the paper, and how to reuse it.

---

## 1. Claims

The paper claims that the development (§"Formalization in Lean") mechanizes the syntax,
type systems and denotational semantics of PAL and Jay, the compiler, and the
correctness theorems of §"Correctness of Compilation", with complete proofs. The
submitted paper states that the proofs assume three laws of floating-point arithmetic;
the artifact needs none of them (see "Floating-point arithmetic" in Section 3). The
table maps each item to the code. All paths are relative to this directory.

| Paper | Lean | File |
|---|---|---|
| PAL syntax and type system (Figs. "Syntax of PAL", "Type system of PAL") | `PAL.Ty`, `PAL.Expr`, `PAL.Wppc`, `PAL.Stmt`, `PAL.Block`, `PAL.Program` (intrinsically typed: the typing rules are the constructors) | `Palladium/PAL/Typing.lean`, `Palladium/PAL/Syntax.lean`, `Palladium/PAL/Operators.lean` |
| PAL denotational semantics (Fig. "Denotational semantics of PAL") | `PAL.Expr.eval`, `PAL.Wppc.eval`, `PAL.Stmt.evalSample`, `PAL.Stmt.evalMarginalize`, `PAL.Stmt.evalCondition`, `PAL.Stmt.eval`, `PAL.Block.eval`, `PAL.Program.eval`; `ℙ⟦Δ⟧` | `Palladium/PAL/Semantics.lean`, `Palladium/PAL/Denote.lean` |
| Jay syntax and type system (Figs. "Syntax of Jay", "Type system of Jay") | `JAX.Ty`, `JAX.Expr`, `JAX.Stmt`, `JAX.Program` (intrinsically typed) | `JAX/Syntax.lean`, `JAX/Operators.lean` |
| Auxiliary functions on shapes: rank, volume, padding, broadcasting, slicing | `Shape.rank`, `Shape.volume`, `Shape.pad`, `Shape.broadcast2` (`σ₁ ⊔ σ₂`), `Shape.Broadcastable` (`σ ≤ σₒ`), `Shape.slice` | `TensorLib/Shape/` |
| Index space `I(σ)` and the index maps *proj*, *flatten*, *unflatten*, *ins* | `Indices σ`, `Indices.proj`, `Indices.flatten`, `Indices.unflatten`, `Indices.ins` | `TensorLib/Indices.lean`, `TensorLib/IndexMaps.lean` |
| Jay denotational semantics (Fig. "Denotational semantics of Jay"): `⟦τ[σ]⟧ = I(σ) → ⟦τ⟧`, `⟦Γ ⊢ e : τ[σ]⟧ : ⟦Γ⟧ → I(σ) → ⟦τ⟧` | `JAX.Ty.Sem`, `JAX.Valuation`, `JAX.Expr.denote`, `JAX.Program.denote` (the `⟦·⟧` notation on Jay syntax) | `JAX/Denote.lean`, `JAX/Semantics.lean` |
| Compilation of PAL to Jay (Fig. "Compiling PAL to Jay"), including `REDUCE(e, m, 0)` | `PAL.Expr.compile`, `PAL.Stmt.compileSample`, `PAL.Stmt.compileMarginalize`, `PAL.Stmt.compileCondition`, `PAL.Stmt.compile`, `PAL.Block.compile`, `PAL.Program.compile` (the `𝒞⟦·⟧` notation); `reduce` | `Palladium/Compiler/Compile/` (see [`Palladium/Compiler/README.md`](Palladium/Compiler/README.md)) |
| Semantic correspondence `θ ~ γ` (parameters) | `JAX.Valuation.RepresentsParams` (`γ ~ θ`) | `Palladium/Compiler/Correctness/WellFormed.lean` |
| Semantic correspondence `δ ~ i` (random environments and indices) | `Environment.toIndices` (the indices of `δ`) | `Palladium/Compiler/Encoding/EnvironmentIndices.lean` |
| Semantic correspondence `μ ~ t` (PMFs and tensors) | `IdxFn.RepresentsPMF` (`f ~ μ`, for the denotation `f : I(𝒞⟦Δ⟧) → Float` of a tensor) | `Palladium/Compiler/Correctness/RepresentsPMF.lean` |
| Theorem "Expression compilation preserves semantics" | `Compiler.Expr.correctness` | `Palladium/Compiler/Correctness/Expr.lean` |
| Theorem "Statement and block compilation preserves semantics" | `Compiler.Stmt.correctness`, `Compiler.Block.correctness` (with the statement cases `Sample.correctness`, `Marginalize.correctness`, `Condition.correctness`) | `Palladium/Compiler/Correctness/{Stmt,Block,Sample,Marginalize,Condition}.lean` |
| Theorem "Program compilation preserves semantics" | `Compiler.Program.correctness` | `Palladium/Compiler/Correctness/Program.lean` |
| Floating-point assumptions (§"Formalization in Lean") | none; see "Floating-point arithmetic" in Section 3 | |

The theorems are stated for the denotational semantics `⟦·⟧` of Jay (`JAX.Expr.denote`)
and a valuation `γ` of the Jay context, as in the paper.

[`Results.lean`](Results.lean) states the paper's three correctness theorems, prints
the axioms each one depends on, and evaluates a few sanity checks.
[`Extras.lean`](Extras.lean) does the same for results beyond the paper's claims: the
statement cases, the evaluation of Jay in a tensor layout with its adequacy theorems and
the program theorem for it (`Program.correctness_eval`), and the specializations of the
whole-program theorems to the `Tabulated` layout.
[`Examples.lean`](Examples.lean) writes two small PAL programs, compiles them to Jay,
and runs both (see [Trying your own programs](#trying-your-own-programs)).

## 2. Getting started

**Requirements.** Linux or macOS with `git` and `curl`, about 10 GB of free disk space
(2.5 GB for the Lean toolchain and about 6 GB for the unpacked Mathlib build),
and an internet connection for downloading the Lean toolchain and the Mathlib build
cache. No other software is needed.

**Versions.** Lean `v4.26.0` (pinned in `lean-toolchain`) and Mathlib at commit
`3ac4753d8b` (pinned in `lakefile.toml` and `lake-manifest.json`). Both are fetched
automatically.

**Installation (about 5 minutes).**

```sh
# 1. Install elan, the Lean toolchain manager (skip if already installed).
curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
  | sh -s -- -y --default-toolchain none
export PATH="$HOME/.elan/bin:$PATH"

# 2. In this directory, download and unpack the prebuilt Mathlib cache (about 6 GB).
lake exe cache get
```

**Build and sanity test (a few minutes).**

```sh
lake build
```

This builds the library, `Results.lean`, `Extras.lean` and `Examples.lean`, re-checking
every proof. On a 16-core
machine it takes about 2 minutes of wall-clock time (about 12 minutes of CPU time). It
ends with `Build completed successfully`. Its messages include the results of the
sanity checks in `Results.lean`, which read:

```
0.000000      -- Float.div0 0 0
5.000000      -- ⟦[1.0, 2.0] + 3.0⟧ at index [1]
5.000000      -- the same entry of the evaluated tensor
0.250000      -- joint sampling of two binary variables with constant weight
0.000000      -- the same with weight 0 (0 / 0 = 0)
0.000000      -- condition false on a NaN mass
```

## 3. Evaluation instructions

**Step 1: re-check the proofs.** `lake build` (Section 2) re-checks every proof in the
development. A proof that does not check makes the build fail.

**Step 2: inspect the main theorems and their axioms.**

```sh
lake build Results Extras
```

prints, for each of the paper's three theorems (`Results.lean`) and each result beyond
them (`Extras.lean`), its statement (`#check`) and the axioms it depends on
(`#print axioms`). Every result depends only on Lean's standard classical basis
`propext`, `Classical.choice`, `Quot.sound`. `sorryAx` does not appear in any list; it
would if any proof were incomplete.

To re-run only this step after a full build, delete the cached results first:

```sh
rm -f .lake/build/lib/lean/Results.* .lake/build/lib/lean/Extras.*
lake build Results Extras
```

**Step 3: replay the proofs in the Lean kernel (about 1 minute).** The artifact
includes [lean4checker](https://github.com/leanprover/lean4checker) (release `v4.26.0`,
matching the Lean version), which reads the compiled `.olean` files and sends every
declaration back to the Lean kernel. This guards against proofs that reach the
environment without passing through the kernel, for example through metaprograms.

```sh
lake build lean4checker
lake exe lean4checker -v Util TensorLib JAX Palladium Results Extras Examples
```

The second command replays every module of the development (76 modules), each on top
of its imports, and prints `replaying <module>` for each. It exits with status `0` and
no other output; a declaration that the kernel rejects makes it print
`lean4checker found a problem in <module>` and exit with a non-zero status. On a
16-core machine it takes about 45 seconds.

Optionally, the replay can start from an empty environment instead of the imports, so
that the kernel also re-checks every declaration of Lean and Mathlib that the
development depends on:

```sh
lake exe lean4checker --fresh Extras
```

`Extras` imports the whole library. This replay is single-threaded and takes
about 5 minutes; it prints nothing and exits with status `0`.

**Step 4: compare the statements with the paper.** Section 4 describes how each
notion of the paper is modelled in Lean. The statements printed in Step 2, together
with the definitions they mention (Section 1), are the claims of the paper's
correctness theorems.

### Floating-point arithmetic

Lean's `Float` is a primitive type backed by IEEE 754 doubles, so its arithmetic is
opaque to the logic, and laws such as `1 · x = x` are not provable about it. The
submitted paper (§"Formalization in Lean") posits three such laws as axioms:
`1 · x = x`, `x · y = y · x`, and `x · 0 = 0`. Since submission, we have revised the
proofs so that they require no posited law about `Float`.
The development thus declares no axioms, and Step 2 shows that every result depends only on
Lean's classical basis. The final version of the paper will reflect this.

## 4. Additional description

### Structure

```
Util/            contexts, environments and their operations, PMFs, macros, Float.div0
TensorLib/       shapes, multi-indices and index maps, tensor layouts and their laws
  Shape/           rank, volume, padding, broadcasting, slicing, reshaping
  Indices.lean     I(σ): proof-carrying multi-indices
  IndexMaps.lean   proj, flatten, unflatten, ins; index-wise sums
  Operations/      the layout interface (Tensor.Operations, Tensor.LawfulOperations)
                   and the Tabulated layout with its instances
  IndexLemmas.lean indexing an operation's result reads its operands at mapped indices
JAX/             Jay: operators, syntax, the denotational semantics ⟦·⟧, the layout
                 evaluation, adequacy (Adequacy.lean), and the jax{…} surface macro
Palladium/PAL/   PAL: operators, typing, syntax, semantics
Palladium/Compiler/
  Compile/         the compiler (see Palladium/Compiler/README.md)
  Encoding/        random environments as tensor indices
  Correctness/     the correctness proofs
Palladium.lean   library root
Results.lean     the paper's theorems, their axioms, sanity checks
Extras.lean      results beyond the paper's claims, their axioms
Examples.lean    two PAL programs compiled to Jay and run
```

### Correspondence with the paper

The paper presents a simplified formalism for exposition (§"Formalization in Lean").
This section lists how each of its notions is modelled.

**Typing.** Syntax is intrinsically typed: a term of `PAL.Expr Δ T` or
`JAX.Expr Γ T` is a typing derivation, and the typing rules of the paper are the
constructors.

**Base types and values.**
- The real numbers of the paper are IEEE 754 doubles (`Float`).
- PAL and Jay share the element types `float`, `nat` and `disc (fin n)` (`Fin n`, with
  `n > 0`); booleans are `Fin 2`, with `0` false and every other value true.
- A random variable `x : Fin n` has the expression type `disc (fin n)`.
- Jay tensor types are `JAX.Ty.tensor σ η`.

**Contexts and environments.**
- Contexts are lists. New random variables are prepended: the paper's `Δ, x⃗ : D⃗` is
  the Lean context `x⃗ ++ Δ`, and `𝒞⟦Δ⟧` (`RandomContext.toShape`) lists the head
  variable first. This matches the paper's `Shp(Δ, x : D) = Shp(D) ++ Shp(Δ)`.
- Statement-level random contexts carry a proof that their names are distinct, and
  static parameters have names outside the compiler's reserved prefix.
- Environments are dependent functions on context membership.

**PAL expressions.**
- `fin i` literals of any `Fin n` and `float r` literals.
- A concrete set of unary operators (`neg`, `not`) and binary operators (`add`, `sub`,
  `mul`, `div`, `and`, `or`, `eq`, `ne`, `lt`, `le`). These accept operands of any
  element types and promote them (`disc < nat < float`).
- `ite` (the paper's `select`) allows branches of different types, joined by promotion.
- The literals `n : N`, `true` and `false` are written `fin` literals.

**Division.** Division in PAL and Jay is `Float.div0`: `0 / 0 = 0`, and IEEE 754
division otherwise. Normalization in `sample` uses the same operation. This makes
normalization total, as in the paper; `x / 0` for `x ≠ 0` is `±∞`.

**PAL statements.**
- `x⃗ ← sample (wpp e)` jointly samples the variables `x⃗`. Its weight expression has
  type `float`, as in rule P-Sample. The normalizing constant sums over all
  valuations of `x⃗`, with the head variable innermost, the order of
  `REDUCE(e, m, 0)`. `Stmt.sample1` is the single-variable case.
- `marginalize x` removes `x` wherever it occurs in the context.
- `condition e` keeps the mass where `e` holds and sets it to `0` elsewhere.
- Sequencing and the empty statement are `seq` and `nil`.
- Branching `x⃗ ← if e then B₁ else B₂` and blocks `B; ret x⃗` follow the paper. Blocks
  may return their variables in a different order from the declared one, and the
  returns of branches and loops are matched with `x⃗` by position and type.
- The loop `x⃗ ← repeat n init {B₀} do {B}` takes its iteration count `n` from an
  expression of type `Fin c` evaluated on the input distribution. Its semantics is the
  mixture, over `k < c`, of the loop unrolled `k` times on the part of the input where
  `n = k`. The paper's `markov` loop with a literal count `n` is the case of a constant
  count expression. The initial block may change the carried context.

**PMFs.** `ℙ⟦Δ⟧` is the type of functions from random environments to `Float`;
sub-probability (non-negativity and total mass at most `1`) is not part of the type.

**Jay expressions.**
- The paper's constants are rank-0 array literals (`array`), and `arange(n)` is the
  array literal `Tensor.arange n`.
- `where(c, t, f)` is `select`, because `where` is a keyword in Lean. Its branches may
  have different element types, joined by promotion.
- `sum(e, n)` is `sum e [n]`.
- `let x = e; e'` is `letIn`.
- `scan` and `reshape` follow the paper.
- Broadcasting appears in binary and ternary operations and in `broadcast_to`, as in
  the paper.
- Constructs beyond the paper: `ones_like`, `sum` over several axes, `squeeze`,
  `transpose` and `comment`, each with an index-wise semantics in the style of the
  paper (`squeeze` reads `ins_n(i, 0)`, `transpose` reads `π⁻¹(i)`).
- A Jay program (`JAX.Program`) has a body of statements (assignments and deletions)
  and a return variable; compiled programs consist of one assignment.

**Jay semantics.**
- `JAX.Expr.denote` is the paper's index-wise semantics, one equation per rule of the
  paper's figure.
- Array literals of Jay hold tensors of a tensor layout `L`, an abstract data
  representation with lawful operations (`Tensor.LawfulOperations`), and `⟦·⟧` reads
  them by indexing. The theorems are proved for every lawful layout. The `Tabulated`
  layout (nested vectors) is an instance, so the hypotheses are satisfiable.
- `JAX.Expr.eval` computes the tensors of Jay expressions in the layout `L`. The
  adequacy theorems `JAX.Expr.eval_toSem` and `JAX.Program.eval_toSem`
  (`JAX/Adequacy.lean`) show that indexing an evaluated tensor at `i` gives `⟦·⟧` at `i`,
  so `Program.correctness_eval` states the program theorem for `eval`.

**Compilation.**
- Expression compilation returns a Jay expression whose shape broadcasts to `𝒞⟦Δ⟧`
  (`Expr.CompileResult`). The correctness predicate reads its denotation at
  `proj(idx(δ))`; the paper's rules C-Param, C-Const and C-Var instead insert
  `broadcast_to` explicitly.
- Statement compilation follows the rules of the paper's figure. Each rule is cited in
  the documentation of its Lean definition.

### Reuse

**Reusable components.** The development is a Lean 4 library under the MIT License,
for Lean `v4.26.0` and Mathlib at commit `3ac4753d8b`. Its definitions and proofs are
usable in other projects:

- **`TensorLib`**: shapes with rank, volume, padding, broadcasting, slicing and
  reshaping; proof-carrying multi-indices `Indices σ` and the index maps *proj*,
  *flatten*, *unflatten* and *ins*; an interface for tensor layouts with their laws
  (`Tensor.Operations`, `Tensor.LawfulOperations`), with the nested-vector `Tabulated`
  layout as an instance; and exact indexing laws for broadcasting, reshaping, squeezing
  and summation in any lawful layout.
- **`JAX`**: Jay, an intrinsically typed core of JAX, with an index-wise denotational
  semantics, an executable evaluator for any lawful layout, adequacy theorems relating
  the two, and the `jax{…}` surface syntax.
- **`Palladium`**: the syntax, type system and semantics of the discrete probabilistic
  language PAL, and a compiler from PAL to Jay whose correctness theorems hold for
  every PAL program and every lawful layout.

Every proof is complete: no result depends on `sorryAx`, the development declares no
axioms, and lean4checker accepts every module (Section 3).

#### Trying your own programs

[`Examples.lean`](Examples.lean) contains two PAL programs: `joint`, which samples two
variables jointly, and `coin`, which uses a parameter, conditioning and
marginalization. For every valuation `δ` of a program's random variables, the file
prints a row with

1. the values of `δ`, newest variable first,
2. the probability `⟦prog⟧ θ δ` that the PAL program assigns to `δ`,
3. the entry at `δ` of the tensor computed by the compiled Jay program `𝒞⟦prog⟧` in
   the `Tabulated` layout,
4. the denotation `⟦𝒞⟦prog⟧⟧ γ` of the compiled program at `δ`.

`lake build Examples` prints the tables:

```
[([0, 0], 0.200000, 0.200000, 0.200000),
 ([0, 1], 0.200000, 0.200000, 0.200000),
 ([1, 0], 0.200000, 0.200000, 0.200000),
 ([1, 1], 0.400000, 0.400000, 0.400000)]
...
[([0], 0.000000, 0.000000, 0.000000), ([1], 0.333333, 0.333333, 0.333333), ([2], 0.383333, 0.383333, 0.383333)]
```

The theorems `joint_correct` and `coin_correct` in the same file are the instances of
`Program.correctness_eval` that guarantee that the columns agree.

To try another program, edit `Examples.lean` and run `lake build Examples`, or open the
file in an editor with the Lean extension, which shows the tables next to the `#eval`
commands. The proofs of `joint_correct` and `coin_correct` apply `Program.correctness_eval`
to the program, so they still check after any change to it. For example, change the weight
`1 + x · y` of `sampleY` or the condition `y ≠ 0` of `conditionY`, or the value `0.3` of
`p` in both `θp` and `pTensor`. PAL has no surface syntax: programs are built from the
constructors of `PAL.Expr`, `PAL.Stmt`, `PAL.Block` and `PAL.Program`
(`Palladium/PAL/Syntax.lean`), and the proofs that variables are in scope and that
names are distinct close with `by decide`.

#### Other uses

- **Writing Jay programs.** The macro `jax{…}` (`JAX/Macros.lean`) provides a surface
  syntax close to JAX for Jay expressions and programs.
- **Other tensor representations.** A new layout `L` needs instances of
  `Tensor.Operations L`, `Tensor.LawfulOperations L`, `Tensor.Operations.OfTabulated L`
  and `Tensor.Operations.LawfulOfTabulated L`. Every theorem then holds for `L`, as
  `Program.correctness_tabulated` in `Extras.lean` illustrates for `Tabulated`.
- **Tensor reasoning.** `TensorLib/IndexLemmas.lean` gives exact indexing laws for
  broadcasting, reshaping, squeezing and summing in any lawful layout, and
  `JAX/Adequacy.lean` relates the evaluation of Jay programs to their index-wise
  semantics.
