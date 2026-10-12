import JAX.Semantics
import TensorLib.IndexLemmas

/-! # Adequacy of the layout evaluation

`Expr.eval` evaluates a Jay expression to a tensor in a layout `L`; `Expr.denote` is the
index-wise denotational semantics `⟦e⟧`. For every lawful layout, indexing the evaluated
tensor at `i` gives `⟦e⟧ γ i` (`Expr.eval_toSem`), and likewise for statements and
programs. Valuations and layout environments are related by `Environment.toValuation`
and `Valuation.toEnvironment`.
-/

namespace Palladium.JAX

open Palladium.Util
open Palladium.TensorLib

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-- The indexing function of a tensor. -/
def Ty.toSem : (T : Ty) → Ty.denote (L := L) T → T.Sem
  | .tensor _ _, t => fun i => Tensor.index t i

/-- The tensor with a given indexing function. -/
def Ty.ofSem : (T : Ty) → T.Sem → Ty.denote (L := L) T
  | .tensor _ _, f => Tensor.ofFn f

/-- The valuation that maps each variable to the indexing function of its tensor. -/
def Environment.toValuation {Γ : Context} (γ : ⟦Γ⟧ (L := L)) : Valuation Γ :=
  fun x T h => Ty.toSem T (γ x T h)

/-- The layout environment that maps each variable to the tensor with its indexing
    function. -/
def Valuation.toEnvironment {Γ : Context} (γ : Valuation Γ) : ⟦Γ⟧ (L := L) :=
  fun x T h => Ty.ofSem T (γ x T h)

theorem Environment.toValuation_cons {Γ : Context} (γ : ⟦Γ⟧ (L := L)) (name : String)
    (T : Ty) (v : Ty.denote (L := L) T) :
    Environment.toValuation (γ.cons name T v) =
      (Environment.toValuation γ).cons name T (Ty.toSem T v) := by
  funext x τ h
  simp only [Environment.toValuation, Environment.cons]
  split
  · next heq => cases heq; rfl
  · rfl

theorem Environment.toValuation_kinsert {Γ : Context} (γ : ⟦Γ⟧ (L := L)) (name : String)
    (T : Ty) (v : Ty.denote (L := L) T) :
    Environment.toValuation (γ.kinsert name T v) =
      (Environment.toValuation γ).kinsert name T (Ty.toSem T v) := by
  funext x τ h
  simp only [Environment.toValuation, Environment.kinsert]
  split
  · next heq => cases heq; rfl
  · rfl

variable [Tensor.LawfulOperations L]

/-- Iterating evaluation and iterating denotation agree. -/
private theorem toSem_iterate {T : Ty} (f : Ty.denote (L := L) T → Ty.denote (L := L) T)
    (g : T.Sem → T.Sem) (hfg : ∀ t, Ty.toSem T (f t) = g (Ty.toSem T t)) :
    (n : Nat) → (t : Ty.denote (L := L) T) →
    Ty.toSem T (Nat.iterate f n t) = Nat.iterate g n (Ty.toSem T t)
  | 0, _ => rfl
  | n + 1, t => by
      simp only [Function.iterate_succ, Function.comp]
      rw [toSem_iterate f g hfg n (f t), hfg]

/-- Adequacy: the indexing function of `e.eval γ` is `⟦e⟧ γ`. -/
theorem Expr.eval_toSem {Γ : Context} {T : Ty} (e : Expr (L := L) Γ T) (γ : ⟦Γ⟧ (L := L)) :
    Ty.toSem T (e.eval γ) = e.denote (Environment.toValuation γ) := by
  induction e with
  | var name T h => rfl
  | array η t => rfl
  | uop op e hv ih =>
    funext i
    simp only [Ty.toSem, JAX.Expr.eval, JAX.Expr.denote]
    rw [← ih γ]
    exact Tensor.Operations.index_map _ _ i
  | bop op a b hv h iha ihb =>
    funext i
    simp only [Ty.toSem, JAX.Expr.eval, JAX.Expr.denote, Tensor.zipWithBroadcast, Tensor.zipWith]
    rw [← iha γ, ← ihb γ]
    simp only [Ty.toSem, ← Tensor.getElem_eq_index]
    erw [Tensor.index_zipWith]
    simp only [Tensor.getElem_eq_index]
    rw [Tensor.index_broadcast, Tensor.index_broadcast]
  | select cond x y h ihc ihx ihy =>
    funext i
    simp only [Ty.toSem, JAX.Expr.eval, JAX.Expr.denote, Tensor.zipWith3Broadcast,
      Tensor.zipWith3]
    rw [← ihc γ, ← ihx γ, ← ihy γ]
    simp only [Ty.toSem, ← Tensor.getElem_eq_index]
    erw [Tensor.index_zipWith3]
    simp only [Tensor.getElem_eq_index]
    rw [Tensor.index_broadcast, Tensor.index_broadcast, Tensor.index_broadcast]
    simp only [← Tensor.getElem_eq_index]
    erw [Tensor.index_map, Tensor.index_map]
  | onesLike arg _ =>
    funext i
    exact Tensor.Operations.index_map _ _ i
  | ones σ =>
    funext i
    exact Tensor.Operations.index_fill _ σ i
  | broadcastTo arg σₒ h ih =>
    funext i
    simp only [Ty.toSem, JAX.Expr.eval, JAX.Expr.denote]
    rw [← ih γ]
    exact Tensor.index_broadcast _ h i
  | reshape arr σₜ h ih =>
    funext i
    simp only [Ty.toSem, JAX.Expr.eval, JAX.Expr.denote]
    rw [← ih γ]
    exact Tensor.index_reshape _ σₜ h i
  | sum arg axes hNodup ih =>
    rename_i σ η
    funext i
    simp only [Ty.toSem, JAX.Expr.eval, JAX.Expr.denote]
    rw [← ih γ]
    cases η with
    | float =>
      simp only [Ty.toSem]
      rw [tensorCastDenoteDtype_index, Tensor.index_sumAxes]
    | nat =>
      simp only [Ty.toSem]
      rw [tensorCastDenoteDtype_index, Tensor.index_sumAxes]
    | disc d =>
      cases d with
      | fin n pos =>
        by_cases hax : axes.length = 0
        · simp only [hax, ↓reduceDIte, Ty.toSem]
          rw [tensorCastDenoteDtype_index, Tensor.index_sumAxes]
        · simp only [hax, ↓reduceDIte, Ty.toSem]
          rw [tensorCastDenoteDtype_index, Tensor.index_sumAxes]
          congr 2
          funext j
          exact Tensor.Operations.index_map _ _ j
  | squeeze arg axis h ih =>
    funext i
    simp only [Ty.toSem, JAX.Expr.eval, JAX.Expr.denote]
    rw [← ih γ]
    exact Tensor.index_squeeze _ _ axis h i
  | transpose arg perm h ih =>
    funext i
    simp only [Ty.toSem, JAX.Expr.eval, JAX.Expr.denote]
    rw [← ih γ]
    exact Tensor.LawfulOperations.index_transpose _ perm h i
  | letIn name value body ihv ihb =>
    simp only [JAX.Expr.eval, JAX.Expr.denote]
    rw [ihb, Environment.toValuation_cons, ihv]
  | comment _ body ih =>
    exact ih γ
  | scan name init n body ihi ihb =>
    simp only [JAX.Expr.eval, JAX.Expr.denote]
    rw [toSem_iterate _ (fun s => body.denote ((Environment.toValuation γ).cons name _ s))
      (fun t => by rw [ihb, Environment.toValuation_cons]) n, ihi]

/-- Adequacy for statements. -/
theorem Stmt.eval_toValuation {Γ Γ' : Context} (s : Stmt (L := L) Γ Γ') (γ : ⟦Γ⟧ (L := L)) :
    Environment.toValuation (s.eval γ) = s.denote (Environment.toValuation γ) := by
  cases s with
  | assign name e =>
    simp only [JAX.Stmt.eval, JAX.Stmt.denote]
    rw [Environment.toValuation_kinsert, Expr.eval_toSem]
  | comment _ => rfl
  | delete name => rfl

/-- Adequacy for statement sequences. -/
theorem Stmts.eval_toValuation {Γ Γ' : Context} :
    (ss : Stmts (L := L) Γ Γ') → (γ : ⟦Γ⟧ (L := L)) →
    Environment.toValuation (ss.eval γ) = ss.denote (Environment.toValuation γ)
  | .nil, _ => rfl
  | .cons s rest, γ => by
      simp only [JAX.Stmts.eval, JAX.Stmts.denote]
      rw [Stmts.eval_toValuation rest, Stmt.eval_toValuation]

/-- Adequacy for programs: the indexing function of `prog.eval γ` is `⟦prog⟧ γ`. -/
theorem Program.eval_toSem {argTy : Context} {retTy : Ty}
    (prog : Program (L := L) argTy retTy) (γ : ⟦argTy⟧ (L := L)) :
    Ty.toSem retTy (prog.eval γ) = prog.denote (Environment.toValuation γ) := by
  cases prog with
  | mk _ body ret h =>
    simp only [JAX.Program.eval, JAX.Program.denote]
    rw [← Stmts.eval_toValuation body γ]
    rfl

end

end Palladium.JAX
