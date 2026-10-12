import Palladium.Compiler.Correctness.Correct
import JAX.Semantics

/-!
# Shared Helper Lemmas for Correctness Proofs

Common proof helpers for the denotation `⟦·⟧` of compiled Jay expressions: transport
along shape equalities and the denotation of the inline enum expression of rule C-Var.
-/

namespace Palladium.Compiler

open Palladium.Util
open Palladium.TensorLib
open Palladium.PAL
open Palladium.JAX
open Shape Tensor

/-- Shape components agree when two `JAX.Ty.tensor` constructors are equal. -/
theorem JAX.Ty.tensor.shape_eq_of_eq {σ σ' : Shape} {η η'}
    (hTy : JAX.Ty.tensor σ η = JAX.Ty.tensor σ' η') : σ' = σ := by
  injection hTy with hσ _
  exact hσ.symm

/-- Dtype components agree when two `JAX.Ty.tensor` constructors are equal. -/
theorem JAX.Ty.tensor.dtype_eq_of_eq {σ σ' : Shape} {η η'}
    (hTy : JAX.Ty.tensor σ η = JAX.Ty.tensor σ' η') : η = η' := by
  injection hTy

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-- The denotation of an expression transported along a shape equality reads the cast
    index. -/
theorem denote_shape_cast {s1 s2 : Shape} {η : DType} {Γ : JAX.Context}
    (h : s1 = s2) (e : JAX.Expr (L := L) Γ (JAX.Ty.tensor s1 η))
    (γ : JAX.Valuation Γ) (i : Indices s2) :
    (h ▸ e).denote γ i = e.denote γ (i.cast h.symm) := by
  subst h; rfl

/-- The denotation of an expression cast along a shape equality reads the cast index. -/
theorem denote_cast_shape {s1 s2 : Shape} {η : DType} {Γ : JAX.Context}
    (h : s1 = s2) (e : JAX.Expr (L := L) Γ (JAX.Ty.tensor s1 η))
    (γ : JAX.Valuation Γ) (i : Indices s2) :
    (cast (congrArg (fun s => JAX.Expr (L := L) Γ (JAX.Ty.tensor s η)) h) e).denote γ i =
      e.denote γ (i.cast h.symm) := by
  subst h; rfl

/-- Applying a function transported along a `JAX.Ty.tensor` equality reads the cast index
    and transports the value. -/
theorem Ty.Sem_cast_apply {σ σ' : Shape} {η η' : DType}
    (h : JAX.Ty.tensor σ η = JAX.Ty.tensor σ' η') (f : Indices σ → ⟦η⟧) (i : Indices σ') :
    (cast (congrArg JAX.Ty.Sem h) f) i =
      cast (congrArg (Denote.denote (α := DType)) (JAX.Ty.tensor.dtype_eq_of_eq h))
        (f (i.cast (JAX.Ty.tensor.shape_eq_of_eq h))) := by
  injection h with hσ hη
  subst hσ hη
  rfl

/-- The denotation of a JAX expression transported along a type equality is transported. -/
theorem denote_cast_ty {Γ : JAX.Context} {τ₁ τ₂ : JAX.Ty} (h : τ₁ = τ₂)
    (e : JAX.Expr (L := L) Γ τ₁) (γ : JAX.Valuation Γ) :
    (cast (congrArg (JAX.Expr (L := L) Γ) h) e).denote γ =
      cast (congrArg JAX.Ty.Sem h) (e.denote γ) := by
  subst h; rfl

/-- The denotation of an expression transported along a type equality is transported. -/
theorem denote_ty_rec {Γ : JAX.Context} {τ₁ τ₂ : JAX.Ty} (h : τ₁ = τ₂)
    (e : JAX.Expr (L := L) Γ τ₁) (γ : JAX.Valuation Γ) :
    (h ▸ e).denote γ = h ▸ e.denote γ := by
  subst h; rfl

/-- Applying an indexing function transported along a `JAX.Ty.tensor` equality reads the
    cast index and transports the value. -/
theorem Ty.Sem_rec_apply {σ σ' : Shape} {η η' : DType}
    (h : JAX.Ty.tensor σ η = JAX.Ty.tensor σ' η') (f : (JAX.Ty.tensor σ η).Sem)
    (i : Indices σ') :
    (h ▸ f : (JAX.Ty.tensor σ' η').Sem) i =
      cast (congrArg (Denote.denote (α := DType)) (JAX.Ty.tensor.dtype_eq_of_eq h))
        (f (i.cast (JAX.Ty.tensor.shape_eq_of_eq h))) := by
  injection h with hσ hη
  subst hσ hη
  rfl

end

/-! ### Inline enum expression -/

section

variable {L : Tensor.Layout} [Tensor.Operations L] [Tensor.OfTabulated L]
  [Tensor.LawfulOfTabulated L]

/-- `arange(n)` at index `v` is `v`. -/
theorem index_arange {n : Nat} (i : Indices [n]) :
    (Tensor.arange (L := L) n)[i] = i.head := by
  rw [Tensor.index_cons, Tensor.arange, Tensor.get_ofTabulated, Tensor.index_nil,
    Tensor.scalar_ofTabulated]
  simp

/-- The values of `unflatten_{[m]}(x)` are `[x]`. -/
theorem Indices.unflatten_singleton_values {m : Nat} (x : Fin (volume [m])) :
    (Indices.unflatten x).values = [x.val] := by
  simp [Indices.unflatten, computeStrides, volume, unravelIndex]

/-- The head entry of `proj(idx(δ))` in the enum shape of the head variable `x` is the value
    of `x` in `δ`. -/
private theorem enum_head_value {x : String} {m : Nat} {hm : 0 < m} {rest : RandomContext}
    (h : ⟨x, Discrete.fin m hm⟩ ∈ (⟨x, .fin m hm⟩ :: rest : RandomContext))
    (bc : (m :: List.replicate rest.keys.length 1) ≤ ↑(⟨x, .fin m hm⟩ :: rest))
    (δ : ⟦⟨x, .fin m hm⟩ :: rest⟧) :
    ((δ.toIndices.proj bc).values.head?.getD 0) = (↑(δ⁅x⁆h) : Fin m).val := by
  have hlen : 0 < (δ.toIndices.proj bc).values.length := by
    rw [(δ.toIndices.proj bc).length_eq]; simp
  rw [List.head?_eq_getElem?, List.getElem?_eq_getElem hlen, Option.getD_some,
    Indices.proj_getElem]
  split
  · next h1 =>
    have := (δ⁅x⁆h).isLt
    simp only [List.getElem_cons_zero] at h1
    omega
  · have hr : (RandomContext.toShape (⟨x, Discrete.fin m hm⟩ :: rest)).rank =
        rest.keys.length + 1 := by
      simp [Shape.rank, RandomContext.toShape_length]
    simp [Environment.toIndices_values, Environment.toIndicesRaw, hr]
    rfl

/-- Helper for the head case of `enumExpr_denote`, with the enum shape as a parameter so
    that it can be substituted. -/
private theorem enumExpr_head_aux {Γ : JAX.Context}
    {x : String} {m : Nat} {hm : 0 < m} {rest : RandomContext}
    (h : ⟨x, Discrete.fin m hm⟩ ∈ (⟨x, .fin m hm⟩ :: rest : RandomContext))
    {es : Shape}
    (hr : [m] ≅ₛ es) (bc : es ≤ ↑(⟨x, .fin m hm⟩ :: rest))
    (hes : es = m :: List.replicate rest.keys.length 1)
    (δ : ⟦⟨x, .fin m hm⟩ :: rest⟧) (γ : JAX.Valuation Γ) :
    (JAX.Expr.reshape (finSupport (L := L) (Γ := Γ) m hm) es hr).denote γ
        (δ.toIndices.proj bc) = ↑(δ⁅x⁆h) := by
  subst hes
  simp only [JAX.Expr.denote, finSupport]
  rw [index_arange]
  apply Fin.ext
  have hflat := Indices.flatten_enum (δ.toIndices.proj bc)
  simp only [Indices.head, Indices.unflatten_singleton_values, List.head_cons]
  show ((δ.toIndices.proj bc).flatten).val = _
  rw [hflat]
  exact enum_head_value h bc δ

omit [Tensor.LawfulOfTabulated L] in
/-- Helper for the tail case of `enumExpr_denote`: the leading index is irrelevant when the
    enum shape broadcasts to the tail context. -/
private theorem enumExpr_tail_aux {Γ : JAX.Context} {n : Nat} {hn : 0 < n}
    {y : String} {m : Nat} {hm : 0 < m} {rest : RandomContext}
    {es₁ es₂ : Shape} (hes : es₁ = es₂)
    (hr₁ : [n] ≅ₛ es₁) (hr₂ : [n] ≅ₛ es₂)
    (bc₁ : es₁ ≤ ↑(⟨y, .fin m hm⟩ :: rest)) (bc₂ : es₂ ≤ ↑rest)
    (δ : ⟦⟨y, .fin m hm⟩ :: rest⟧) (γ : JAX.Valuation Γ) :
    (JAX.Expr.reshape (finSupport (L := L) (Γ := Γ) n hn) es₁ hr₁).denote γ
        (δ.toIndices.proj bc₁) =
      (JAX.Expr.reshape (finSupport (L := L) (Γ := Γ) n hn) es₂ hr₂).denote γ
        (δ.tail.toIndices.proj bc₂) := by
  subst hes
  rw [Indices.proj_cons bc₁ bc₂]
  rfl

/-- The inline enum expression of a variable `x`, read at the indices of `δ` projected to
    the enum shape, gives the value of `x` in `δ`. -/
theorem enumExpr_denote
    {Δ : RandomContext}
    {Γ : JAX.Context} {x : String} {n : Nat} {hn : 0 < n}
    (h : ⟨x, .fin n hn⟩ ∈ Δ)
    (hnd : Δ.keys.Nodup)
    (δ : ⟦Δ⟧)
    (γ : JAX.Valuation Γ) :
    (enumExpr (L := L) h hnd).denote γ (δ.toIndices.proj (enumShape_broadcastable h hnd)) =
      ↑(δ⁅x⁆h) := by
  induction Δ with
  | nil => exact nomatch h
  | cons hd rest ih =>
    match hd with
    | ⟨y, .fin m hm⟩ =>
    by_cases hxy : x = y
    · subst hxy
      have hmn : m = n := by
        cases h with
        | head => rfl
        | tail _ h' => exact absurd (List.mem_keys.mpr ⟨_, h'⟩) (List.nodup_cons.mp hnd).1
      subst hmn
      exact enumExpr_head_aux h (enumShape_reshapeable h hnd)
        (enumShape_broadcastable h hnd) enumShape_self δ γ
    · have h_orig : ⟨x, .fin n hn⟩ ∈ rest := Context.mem_cons_ne hxy h
      have hnd_rest : rest.keys.Nodup := (List.nodup_cons.mp hnd).2
      exact (enumExpr_tail_aux (enumShape_ne hxy)
        (enumShape_reshapeable h hnd) (enumShape_reshapeable h_orig hnd_rest)
        (enumShape_broadcastable h hnd) (enumShape_broadcastable h_orig hnd_rest)
        δ γ).trans (ih h_orig hnd_rest δ.tail)

end

/-- `Ty.toBool` of the boolean NOT of a `Fin 2` value equals the boolean NOT of `Ty.toBool`. -/
theorem PAL.Ty.toBool_not_eq (v : ⟦PAL.Ty.bool⟧) :
    PAL.Ty.toBool .bool (↑(!PAL.Ty.toBool .bool v) : ⟦PAL.Ty.bool⟧) =
    !(PAL.Ty.toBool .bool v) := by
  revert v; decide

end Palladium.Compiler

