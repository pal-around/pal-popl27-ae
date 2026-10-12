import JAX.Syntax
import Util.Float
import JAX.Denote
import TensorLib.Notations
import TensorLib.IndexMaps
import Util.Macros

namespace Palladium.JAX

open Palladium.Util
open Palladium.TensorLib
open Shape

/-!
Semantics of Jay (the JAX target language).

* `Expr.denote`, `Stmt.denote`, `Stmts.denote` and `Program.denote` give the index-wise
  denotational semantics of the paper: an expression denotes, for each valuation, the
  indexing function of a tensor. The `⟦·⟧` notation on Jay syntax refers to these.
* `Expr.eval`, `Stmt.eval`, `Stmts.eval` and `Program.eval` evaluate the same syntax to
  tensors in a layout `L` (any `L` with `[Tensor.Operations L]`). `JAX.Adequacy` proves
  that indexing their result agrees with the denotational semantics.

All functions are total: the intrinsically typed syntax guarantees that shape
operations succeed.
-/

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-! # JAX Unary operator evaluation (scalar level) -/

/-- Evaluate a JAX unary operator at the scalar level given a validity proof. -/
def UOp.Valid.eval {op : UOp} {η ηₒ : DType}
    (h : UOp.Valid op η ηₒ) (v : ⟦η⟧) : ⟦ηₒ⟧ :=
  match h with
  | .neg_f => -v
  | .neg_n => -(Float.ofNat v)
  | .neg_d => -(Float.ofNat v.val)
  | .not_f => ↑(!(↑v : Bool))
  | .not_n => ↑(!DType.toBool _ v)
  | .not_d => ↑(!DType.toBool _ v)

/-! # JAX Binary operator evaluation (scalar level) -/

/-- Evaluate a JAX binary operator at the scalar level given a validity proof. -/
def BOp.Valid.eval {op : BOp} {η₁ η₂ ηₒ : DType}
    (h : BOp.Valid op η₁ η₂ ηₒ) (a : ⟦η₁⟧) (b : ⟦η₂⟧) : ⟦ηₒ⟧ :=
  match h with
  -- add
  | .add_ff => a + b
  | .add_fn => a + Float.ofNat b
  | .add_fd => a + Float.ofNat b.val
  | .add_nf => Float.ofNat a + b
  | .add_nn => a + b
  | .add_nd => a + b.val
  | .add_df => Float.ofNat a.val + b
  | .add_dn => a.val + b
  | .add_dd => a.val + b.val
  -- sub
  | .sub_ff => a - b
  | .sub_fn => a - Float.ofNat b
  | .sub_fd => a - Float.ofNat b.val
  | .sub_nf => Float.ofNat a - b
  | .sub_nn => a - b
  | .sub_nd => a - b.val
  | .sub_df => Float.ofNat a.val - b
  | .sub_dn => a.val - b
  | .sub_dd => ⟨a.val - b.val, by omega⟩
  -- mul
  | .mul_ff => a * b
  | .mul_fn => a * Float.ofNat b
  | .mul_fd => a * Float.ofNat b.val
  | .mul_nf => Float.ofNat a * b
  | .mul_nn => a * b
  | .mul_nd => a * b.val
  | .mul_df => Float.ofNat a.val * b
  | .mul_dn => a.val * b
  | .mul_dd => ⟨a.val * b.val, Nat.mul_lt_mul_of_lt_of_lt a.isLt b.isLt⟩
  -- div
  | .div_ff => Float.div0 a b
  | .div_fn => Float.div0 a (Float.ofNat b)
  | .div_fd => Float.div0 a (Float.ofNat b.val)
  | .div_nf => Float.div0 (Float.ofNat a) b
  | .div_nn => Float.div0 (Float.ofNat a) (Float.ofNat b)
  | .div_nd => Float.div0 (Float.ofNat a) (Float.ofNat b.val)
  | .div_df => Float.div0 (Float.ofNat a.val) b
  | .div_dn => Float.div0 (Float.ofNat a.val) (Float.ofNat b)
  | .div_dd => Float.div0 (Float.ofNat a.val) (Float.ofNat b.val)
  -- and
  | .and_ff => ↑((↑a : Bool) && (↑b : Bool))
  | .and_fn => ↑((↑a : Bool) && DType.toBool _ b)
  | .and_fd => ↑((↑a : Bool) && DType.toBool _ b)
  | .and_nf => ↑(DType.toBool _ a && (↑b : Bool))
  | .and_nn => ↑(DType.toBool _ a && DType.toBool _ b)
  | .and_nd => ↑(DType.toBool _ a && DType.toBool _ b)
  | .and_df => ↑(DType.toBool _ a && (↑b : Bool))
  | .and_dn => ↑(DType.toBool _ a && DType.toBool _ b)
  | .and_dd => ↑(DType.toBool _ a && DType.toBool _ b)
  -- or
  | .or_ff => ↑((↑a : Bool) || (↑b : Bool))
  | .or_fn => ↑((↑a : Bool) || DType.toBool _ b)
  | .or_fd => ↑((↑a : Bool) || DType.toBool _ b)
  | .or_nf => ↑(DType.toBool _ a || (↑b : Bool))
  | .or_nn => ↑(DType.toBool _ a || DType.toBool _ b)
  | .or_nd => ↑(DType.toBool _ a || DType.toBool _ b)
  | .or_df => ↑(DType.toBool _ a || (↑b : Bool))
  | .or_dn => ↑(DType.toBool _ a || DType.toBool _ b)
  | .or_dd => ↑(DType.toBool _ a || DType.toBool _ b)
  -- eq
  | .eq_ff => ↑(a == b)
  | .eq_fn => ↑(a == Float.ofNat b)
  | .eq_fd => ↑(a == Float.ofNat b.val)
  | .eq_nf => ↑(Float.ofNat a == b)
  | .eq_nn => ↑(a == b)
  | .eq_nd => ↑(a == b.val)
  | .eq_df => ↑(Float.ofNat a.val == b)
  | .eq_dn => ↑(a.val == b)
  | .eq_dd => ↑(a.val == b.val)
  -- ne
  | .ne_ff => ↑(a != b)
  | .ne_fn => ↑(a != Float.ofNat b)
  | .ne_fd => ↑(a != Float.ofNat b.val)
  | .ne_nf => ↑(Float.ofNat a != b)
  | .ne_nn => ↑(a != b)
  | .ne_nd => ↑(a != b.val)
  | .ne_df => ↑(Float.ofNat a.val != b)
  | .ne_dn => ↑(a.val != b)
  | .ne_dd => ↑(a.val != b.val)
  -- lt
  | .lt_ff => ↑(decide (a < b))
  | .lt_fn => ↑(decide (a < Float.ofNat b))
  | .lt_fd => ↑(decide (a < Float.ofNat b.val))
  | .lt_nf => ↑(decide (Float.ofNat a < b))
  | .lt_nn => ↑(decide (a < b))
  | .lt_nd => ↑(decide (a < b.val))
  | .lt_df => ↑(decide (Float.ofNat a.val < b))
  | .lt_dn => ↑(decide (a.val < b))
  | .lt_dd => ↑(decide (a.val < b.val))
  -- le
  | .le_ff => ↑(decide (a ≤ b))
  | .le_fn => ↑(decide (a ≤ Float.ofNat b))
  | .le_fd => ↑(decide (a ≤ Float.ofNat b.val))
  | .le_nf => ↑(decide (Float.ofNat a ≤ b))
  | .le_nn => ↑(decide (a ≤ b))
  | .le_nd => ↑(decide (a ≤ b.val))
  | .le_df => ↑(decide (Float.ofNat a.val ≤ b))
  | .le_dn => ↑(decide (a.val ≤ b))
  | .le_dd => ↑(decide (a.val ≤ b.val))
  -- gt
  | .gt_ff => ↑(decide (a > b))
  | .gt_fn => ↑(decide (a > Float.ofNat b))
  | .gt_fd => ↑(decide (a > Float.ofNat b.val))
  | .gt_nf => ↑(decide (Float.ofNat a > b))
  | .gt_nn => ↑(decide (a > b))
  | .gt_nd => ↑(decide (a > b.val))
  | .gt_df => ↑(decide (Float.ofNat a.val > b))
  | .gt_dn => ↑(decide (a.val > b))
  | .gt_dd => ↑(decide (a.val > b.val))
  -- ge
  | .ge_ff => ↑(decide (a ≥ b))
  | .ge_fn => ↑(decide (a ≥ Float.ofNat b))
  | .ge_fd => ↑(decide (a ≥ Float.ofNat b.val))
  | .ge_nf => ↑(decide (Float.ofNat a ≥ b))
  | .ge_nn => ↑(decide (a ≥ b))
  | .ge_nd => ↑(decide (a ≥ b.val))
  | .ge_df => ↑(decide (Float.ofNat a.val ≥ b))
  | .ge_dn => ↑(decide (a.val ≥ b))
  | .ge_dd => ↑(decide (a.val ≥ b.val))

/-- Cast a tensor along an equality of dtype tags (denotation-level). -/
abbrev tensorCastDenoteDtype {σ : Shape} {η η' : DType}
    (h : η = η') (t : Tensor (L := L) σ ⟦η⟧) : Tensor (L := L) σ ⟦η'⟧ :=
  cast (congrArg (fun d => Tensor (L := L) σ ⟦d⟧) h) t

/-- General case: transport the scalar across `⟦η⟧ = ⟦η'⟧`. -/
theorem tensorCastDenoteDtype_index {σ : Shape} {η η' : DType} (h : η = η')
    (t : Tensor (L := L) σ ⟦η⟧) (idx : Indices σ) :
    Tensor.index (tensorCastDenoteDtype h t) idx =
      cast (congrArg (Denote.denote (α := DType)) h) (Tensor.index t idx) := by
  rcases h with rfl
  simp [tensorCastDenoteDtype]

/- Evaluate a JAX expression in a layout-parameterized environment. -/
def Expr.eval (e : Expr (L := L) ctx T) (γ : ⟦ctx⟧ (L := L)) :
    Ty.denote (L := L) T :=
  match e with
  | Expr.var name τ h =>
      γ⁅name⁆h
  | Expr.array _ t =>
      t
  | Expr.uop _ e hv =>
      let v := e.eval γ
      v.map hv.eval
  | Expr.bop (σ₁ := σ₁) (σ₂ := σ₂) _ a b hv (h : σ₁ ⋈ σ₂) =>
      let va := a.eval γ
      let vb := b.eval γ
      va.zipWithBroadcast vb (σ₁ ⊔ σ₂ ‖ h) hv.eval
        (Shape.Broadcastable.of_broadcastCompatible_left h)
        (Shape.Broadcastable.of_broadcastCompatible_right h)
  | Expr.select (σ₃ := σ₃) (σ₁ := σ₁) (σ₂ := σ₂) (η₁ := η₁) (η₂ := η₂)
      cond x y (h : ⋈ₙ #v[σ₃, σ₁, σ₂]) =>
      let vc := cond.eval γ
      let vx := x.eval γ
      let vy := y.eval γ
      let dtypeTo := η₁ ⊔ η₂
      let vx' : Tensor (L := L) σ₁ ⟦dtypeTo⟧ :=
        vx.map (fun v => DType.cast η₁ dtypeTo v (DType.Castable.promote_left η₁ η₂))
      let vy' : Tensor (L := L) σ₂ ⟦dtypeTo⟧ :=
        vy.map (fun v => DType.cast η₂ dtypeTo v (DType.Castable.promote_right η₁ η₂))
      vc.zipWith3Broadcast vx' vy' (⊔ₙ #v[σ₃, σ₁, σ₂] ‖ h)
        (fun (c : Fin 2) t e => if ↑c then t else e)
        (Shape.Broadcastable.of_broadcastNCompatible h 0)
        (Shape.Broadcastable.of_broadcastNCompatible h 1)
        (Shape.Broadcastable.of_broadcastNCompatible h 2)
  | Expr.onesLike (σ := σ) arg =>
      let v := arg.eval γ
      let data : L σ Float := v.map (fun _ => 1.0)
      data
  | Expr.ones σ =>
      Tensor.fill 1.0 σ
  | Expr.broadcastTo arg σₒ h =>
      let v := arg.eval γ
      Tensor.broadcast v σₒ h
  | Expr.reshape arr σₜ h =>
      let arrVal := arr.eval γ
      arrVal.reshape σₜ h
  | Expr.sum (σ := σ) (η := η) arg axes hNodup =>
      match η, arg.eval γ with
      | .float, v =>
          tensorCastDenoteDtype (DType.sumOutMany_float axes.length).symm
            (Tensor.sumAxes (· + ·) (0 : Float) v axes hNodup)
      | .nat, v =>
          tensorCastDenoteDtype (DType.sumOutMany_nat axes.length).symm
            (Tensor.sumAxes Nat.add 0 v axes hNodup)
      | .disc (.fin n pos), v =>
          if hax : axes.length = 0 then
            tensorCastDenoteDtype (by rw [hax]; rfl :
                DType.disc (.fin n pos) = (DType.disc (.fin n pos)).sumOutMany axes.length)
              (Tensor.sumAxes (DType.add (.disc (.fin n pos))) (DType.zero (.disc (.fin n pos))) v axes hNodup)
          else
            tensorCastDenoteDtype (DType.sumOutMany_disc_fin_eq_nat hax).symm
              (Tensor.sumAxes Nat.add 0 (v.map (·.val)) axes hNodup)
  | Expr.squeeze (σ := σ) (η := _) arg axis h =>
      let v := arg.eval γ
      v.squeeze axis h
  | Expr.transpose (σ := σ) (η := _) arg perm h =>
      let v := arg.eval γ
      v.transpose perm h
  | Expr.letIn name value body =>
      let v := value.eval γ
      body.eval (γ.cons name _ v)
  | Expr.comment _ body => body.eval γ
  | Expr.scan name init n body =>
      let t₀ := init.eval γ
      let f := fun t => body.eval (γ.cons name _ t)
      Nat.iterate f n t₀

/- Evaluate a single statement in a layout-parameterized environment. -/
def Stmt.eval (s : Stmt (L := L) ctx ctx') (γ : ⟦ctx⟧ (L := L)) :
    ⟦ctx'⟧ (L := L) :=
  match s with
  | Stmt.assign (name := name) (T := T) e =>
      γ.kinsert name T (e.eval γ)
  | Stmt.comment _ => γ
  | Stmt.delete name =>
      γ.kerase name

/- Evaluate a list of statements in a layout-parameterized environment. -/
def Stmts.eval (ss : Stmts (L := L) ctx ctx') (γ : ⟦ctx⟧ (L := L)) :
    ⟦ctx'⟧ (L := L) :=
  match ss with
  | Stmts.nil => γ
  | Stmts.cons s rest =>
    γ |> s.eval |> rest.eval

/- Evaluate a program by executing its body and returning the value of the return variable. -/
def Program.eval {argTy : Context} {retTy : Ty} (prog : Program (L := L) argTy retTy)
    (γ : ⟦argTy⟧ (L := L)) :
    Ty.denote (L := L) retTy :=
  match prog with
  | mk _ body ret h =>
      let γ' := body.eval γ
      γ'⁅ret⁆h


/-- Evaluating appended statement sequences is the same as sequential evaluation. -/
def Stmts.eval_append : (s1 : Stmts (L := L) ctx ctx') → (s2 : Stmts (L := L) ctx' ctx'') →
    (γ : ⟦ctx⟧ (L := L)) →
    (s1.append s2).eval γ = s2.eval (s1.eval γ)
  | .nil, _, _ => rfl
  | .cons s ss, s2, γ => γ |> s.eval |> Stmts.eval_append ss s2

/-! # Index-wise denotational semantics

`⟦Γ ⊢ e : τ[σ]⟧ : ⟦Γ⟧ → I(σ) → ⟦τ⟧` (paper, Fig. "Denotational semantics of Jay"). Each
rule defines the entry of the result at an output index `i` from entries of the operands
at indices obtained with the index maps `proj`, `flatten`/`unflatten` and `ins`.
`Expr.eval` above computes the same tensors in a layout `L`; `JAX.Adequacy` proves that
indexing `e.eval γ` at `i` gives `⟦e⟧ γ i`. -/

/-- Transport a scalar along an equality of dtype tags. -/
abbrev castDenoteDtype {η η' : DType} (h : η = η') (v : ⟦η⟧) : ⟦η'⟧ :=
  cast (congrArg (Denote.denote (α := DType)) h) v

/-- The index-wise denotation of a JAX expression. -/
def Expr.denote (e : Expr (L := L) ctx T) (γ : Valuation ctx) : T.Sem :=
  match e with
  -- ⟦x⟧ γ [i] = γ(x)[i]
  | Expr.var name _ h => γ⁅name⁆h
  -- ⟦c⟧ γ [i] = ⟦c⟧: an array literal reads its data; a scalar literal is a rank-0 array.
  | Expr.array _ t => fun i => t[i]
  -- ⟦◁ e⟧ γ [i] = ⟦◁⟧(⟦e⟧ γ [i])
  | Expr.uop _ e hv => fun i => hv.eval (e.denote γ i)
  -- ⟦e₁ ⊚ e₂⟧ γ [i] = ⟦⊚⟧(⟦e₁⟧ γ [proj(i)], ⟦e₂⟧ γ [proj(i)])
  | Expr.bop _ a b hv h => fun i =>
      hv.eval
        (a.denote γ (i.proj (Shape.Broadcastable.of_broadcastCompatible_left h)))
        (b.denote γ (i.proj (Shape.Broadcastable.of_broadcastCompatible_right h)))
  -- ⟦where(c, t, f)⟧ γ [i] = if ⟦c⟧ γ [proj(i)] then ⟦t⟧ γ [proj(i)] else ⟦f⟧ γ [proj(i)],
  -- with both branches cast to the promoted dtype.
  | Expr.select (η₁ := η₁) (η₂ := η₂) cond x y h => fun i =>
      if ↑(cond.denote γ (i.proj (Shape.Broadcastable.of_broadcastNCompatible h 0))) then
        DType.cast η₁ (η₁ ⊔ η₂) (x.denote γ (i.proj (Shape.Broadcastable.of_broadcastNCompatible h 1)))
          (DType.Castable.promote_left η₁ η₂)
      else
        DType.cast η₂ (η₁ ⊔ η₂) (y.denote γ (i.proj (Shape.Broadcastable.of_broadcastNCompatible h 2)))
          (DType.Castable.promote_right η₁ η₂)
  -- ⟦ones_like(e)⟧ γ [i] = 1.0
  | Expr.onesLike _ => fun _ => 1.0
  -- ⟦ones(σ)⟧ γ [i] = 1.0
  | Expr.ones _ => fun _ => 1.0
  -- ⟦broadcast_to(e, σₒ)⟧ γ [i] = ⟦e⟧ γ [proj(i)]
  | Expr.broadcastTo arg _ h => fun i => arg.denote γ (i.proj h)
  -- ⟦reshape(e, σₒ)⟧ γ [i] = ⟦e⟧ γ [unflatten_σ(flatten_σₒ(i))]
  | Expr.reshape arr _ h => fun i => arr.denote γ (Indices.unflatten (i.flatten.cast h.symm))
  -- ⟦sum(e, n⃗)⟧ γ [i] = Σ_{j⃗} ⟦e⟧ γ [ins_n⃗(i, j⃗)], summed in the dtype of `eval`.
  | Expr.sum (η := η) arg axes hNodup =>
      match η, arg.denote γ with
      | .float, f => fun i =>
          castDenoteDtype (DType.sumOutMany_float axes.length).symm
            (IdxFn.sumAxes (· + ·) (0 : Float) f axes hNodup i)
      | .nat, f => fun i =>
          castDenoteDtype (DType.sumOutMany_nat axes.length).symm
            (IdxFn.sumAxes Nat.add 0 f axes hNodup i)
      | .disc (.fin n pos), f =>
          if hax : axes.length = 0 then fun i =>
            castDenoteDtype (by rw [hax]; rfl :
                DType.disc (.fin n pos) = (DType.disc (.fin n pos)).sumOutMany axes.length)
              (IdxFn.sumAxes (DType.add (.disc (.fin n pos))) (DType.zero (.disc (.fin n pos)))
                f axes hNodup i)
          else fun i =>
            castDenoteDtype (DType.sumOutMany_disc_fin_eq_nat hax).symm
              (IdxFn.sumAxes Nat.add 0 (fun j => (f j).val) axes hNodup i)
  -- ⟦squeeze(e, n)⟧ γ [i] = ⟦e⟧ γ [ins_n(i, 0)]
  | Expr.squeeze arg axis h => fun i =>
      arg.denote γ (i.ins axis 0 (by unfold Shape.Squeezable at h; omega))
  -- ⟦transpose(e, π)⟧ γ [i] = ⟦e⟧ γ [π⁻¹(i)]
  | Expr.transpose arg _ _ => fun i => arg.denote γ i.invPermute
  -- ⟦let x = e; e'⟧ γ = ⟦e'⟧ (γ[x ↦ ⟦e⟧ γ])
  | Expr.letIn name value body => body.denote (γ.cons name _ (value.denote γ))
  -- ⟦comment s e⟧ γ = ⟦e⟧ γ
  | Expr.comment _ body => body.denote γ
  -- ⟦scan x ← e for n in e'⟧ γ = fⁿ(⟦e⟧ γ), where f(t) = ⟦e'⟧ (γ[x ↦ t])
  | Expr.scan name init n body =>
      Nat.iterate (fun t => body.denote (γ.cons name _ t)) n (init.denote γ)

/-- The denotation of a statement: a valuation transformer. -/
def Stmt.denote (s : Stmt (L := L) ctx ctx') (γ : Valuation ctx) : Valuation ctx' :=
  match s with
  | Stmt.assign (name := name) (T := T) e => γ.kinsert name T (e.denote γ)
  | Stmt.comment _ => γ
  | Stmt.delete name => γ.kerase name

/-- The denotation of a statement sequence: the composition of its statements. -/
def Stmts.denote (ss : Stmts (L := L) ctx ctx') (γ : Valuation ctx) : Valuation ctx' :=
  match ss with
  | Stmts.nil => γ
  | Stmts.cons s rest => γ |> s.denote |> rest.denote

/-- The denotation of a program: run the body and read the return variable. -/
def Program.denote {argTy : Context} {retTy : Ty} (prog : Program (L := L) argTy retTy)
    (γ : Valuation argTy) : retTy.Sem :=
  match prog with
  | mk _ body ret h => (body.denote γ)⁅ret⁆h

/-! # Denote instances for JAX AST nodes -/

/-- `⟦e⟧ : ⟦Γ⟧ → ⟦T⟧`. A JAX expression denotes a function from valuations to indexing
    functions. -/
@[reducible] instance : Util.Denote (Expr (L := L) Γ T) where
  Sem := Valuation Γ → T.Sem
  denote e := e.denote

/-- `⟦s⟧ : ⟦Γ⟧ → ⟦Γ'⟧`. A JAX statement denotes a valuation transformer. -/
@[reducible] instance : Util.Denote (Stmt (L := L) Γ Γ') where
  Sem := Valuation Γ → Valuation Γ'
  denote s := s.denote

/-- `⟦ss⟧ : ⟦Γ⟧ → ⟦Γ'⟧`. A JAX statement sequence denotes a valuation transformer. -/
@[reducible] instance : Util.Denote (Stmts (L := L) Γ Γ') where
  Sem := Valuation Γ → Valuation Γ'
  denote ss := ss.denote

/-- `⟦p⟧ : ⟦Γ⟧ → ⟦T⟧`. A JAX program denotes a function from valuations of its parameters
    to the indexing function of its result. -/
@[reducible] instance : Util.Denote (Program (L := L) argTy retTy) where
  Sem := Valuation argTy → retTy.Sem
  denote prog := prog.denote

end

end Palladium.JAX
