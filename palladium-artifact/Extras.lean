/-
Results beyond the paper's claims.

This file states results of the development that the paper does not claim, and prints
the axioms each one depends on (`lake build Extras`). `Results.lean` states the paper's
theorems.
-/

import Palladium

open Palladium.Compiler

/-!
### Program correctness for the layout evaluation of Jay

`Program.correctness_eval` states the theorem "Program compilation preserves semantics"
for the evaluation of the compiled program in a tensor layout `L`. It follows from
`Program.correctness` and the adequacy theorem below.
-/
#check @Program.correctness_eval
#print axioms Program.correctness_eval

/-!
### Statement cases with dedicated compilation rules

The cases `sample`, `marginalize` and `condition` of the theorem "Statement and block
compilation preserves semantics" (rules C-Sample, C-Marg, C-Condition).
-/
#check @Sample.correctness
#print axioms Sample.correctness

#check @Marginalize.correctness
#print axioms Marginalize.correctness

#check @Condition.correctness
#print axioms Condition.correctness

/-!
### Adequacy of the layout evaluation of Jay

For every lawful layout, indexing the tensor computed by `JAX.Expr.eval` at `i` gives
the index-wise denotation `⟦e⟧ γ i` (Fig. "Denotational semantics of Jay"), so the two
readings coincide.
-/
#check @Palladium.JAX.Expr.eval_toSem
#print axioms Palladium.JAX.Expr.eval_toSem

#check @Palladium.JAX.Program.eval_toSem
#print axioms Palladium.JAX.Program.eval_toSem

/-!
### The layout hypotheses are inhabited (non-vacuity)

The theorems are polymorphic over an abstract lawful layout `L`. The concrete
`Tabulated` layout (`TensorLib/Operations/{Instances,LawfulInstances}.lean`) is a model
of all four typeclass hypotheses, so the results are not vacuously quantified over an
empty class. Specializing the whole-program theorems to `Tabulated` discharges every
hypothesis; the `#print axioms` output below is the same set, which confirms that the
instance proofs introduce no further axioms.
-/

open Palladium.TensorLib (Tabulated)

/-- Whole-program correctness specialized to the concrete `Tabulated` layout. -/
def Program.correctness_tabulated :=
  @Program.correctness Tabulated inferInstance inferInstance inferInstance inferInstance
#print axioms Program.correctness_tabulated

/-- Whole-program correctness for the layout evaluation, specialized to the `Tabulated`
    layout. -/
def Program.correctness_eval_tabulated :=
  @Program.correctness_eval Tabulated inferInstance inferInstance inferInstance inferInstance
#print axioms Program.correctness_eval_tabulated
