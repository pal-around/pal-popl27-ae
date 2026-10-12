import Palladium.Compiler.Compile.Naming
import Palladium.Compiler.Encoding.ContextShape
import JAX.Syntax
import JAX.Macros
import JAX.Semantics
import TensorLib.TensorLemmas

namespace Palladium.Compiler

open Palladium.Util
open Palladium.TensorLib
open Palladium.PAL

/-! ### Type and Operator Compilation -/

/-- Compile a PAL type to a JAX element type (DType). -/
def Ty.compile : PAL.Ty → DType
  | .float => .float
  | .nat => .nat
  | .disc d => .disc d

/-- PAL types compile to JAX element types (DType). -/
@[reducible] instance : Util.Compile PAL.Ty where
  Target := DType
  compile := Ty.compile

/-- Convert a PAL denoted value to the corresponding DType denoted value.
    This is computationally the identity since `Ty.compile` preserves `denote`. -/
@[reducible] def PAL.Ty.toDType : (T : PAL.Ty) → T.denote → ⟦𝒞⟦T⟧⟧
  | .float, v => v
  | .nat, v => v
  | .disc (.fin _ _), v => v

instance (T : PAL.Ty) : Coe T.denote ⟦𝒞⟦T⟧⟧ where
  coe := PAL.Ty.toDType T

/-- Ty.compile commutes with type promotion. -/
theorem Ty.compile_promote (T₁ T₂ : PAL.Ty) :
    𝒞⟦T₁ ⊔ T₂⟧ = 𝒞⟦T₁⟧ ⊔ 𝒞⟦T₂⟧ := by
  cases T₁ with
  | float => cases T₂ <;> (first | (rename_i d; cases d) | skip) <;> rfl
  | nat => cases T₂ <;> (first | (rename_i d; cases d) | skip) <;> rfl
  | disc da => cases da with | fin m _ =>
    cases T₂ with
    | float => rfl
    | nat => rfl
    | disc db => cases db with | fin n _ => rfl

/-- Map PAL unary operators to JAX unary operators. -/
def UOp.compile : UOp → JAX.UOp
  | UOp.neg => JAX.UOp.neg
  | UOp.not => JAX.UOp.not

/-- Map PAL binary operators to JAX binary operators. -/
def BOp.compile : BOp → JAX.BOp
  | BOp.add => JAX.BOp.add
  | BOp.sub => JAX.BOp.sub
  | BOp.mul => JAX.BOp.mul
  | BOp.div => JAX.BOp.div
  | BOp.and => JAX.BOp.and
  | BOp.or => JAX.BOp.or
  | BOp.eq => JAX.BOp.eq
  | BOp.ne => JAX.BOp.ne
  | BOp.lt => JAX.BOp.lt
  | BOp.le => JAX.BOp.le

/-- Compile a PAL unary validity proof to a JAX validity proof. -/
def UOp.Valid.compile (h : PAL.UOp.Valid op T Tₒ)
    {η : DType} : JAX.UOp.Valid (UOp.compile op) η 𝒞⟦Tₒ⟧ :=
  match h with
  | .ofNeg h => match h with | .f | .n | .d => .neg
  | .ofNot h => match h with | .f | .n | .d => .not

/-- Compile a PAL binary validity proof to a JAX validity proof with precise types. -/
def BOp.Valid.compile (h : PAL.BOp.Valid op T₁ T₂ Tₒ) :
    JAX.BOp.Valid (BOp.compile op) 𝒞⟦T₁⟧ 𝒞⟦T₂⟧ 𝒞⟦Tₒ⟧ :=
  match h with
  | .ofAdd h => match h with
    | .ff => .add_ff | .fn => .add_fn | .fd => .add_fd
    | .nf => .add_nf | .nn => .add_nn | .nd => .add_nd
    | .df => .add_df | .dn => .add_dn | .dd => .add_dd
  | .ofSub h => match h with
    | .ff => .sub_ff | .fn => .sub_fn | .fd => .sub_fd
    | .nf => .sub_nf | .nn => .sub_nn | .nd => .sub_nd
    | .df => .sub_df | .dn => .sub_dn | .dd => .sub_dd
  | .ofMul h => match h with
    | .ff => .mul_ff | .fn => .mul_fn | .fd => .mul_fd
    | .nf => .mul_nf | .nn => .mul_nn | .nd => .mul_nd
    | .df => .mul_df | .dn => .mul_dn | .dd => .mul_dd
  | .ofDiv h => match h with
    | .ff => .div_ff | .fn => .div_fn | .fd => .div_fd
    | .nf => .div_nf | .nn => .div_nn | .nd => .div_nd
    | .df => .div_df | .dn => .div_dn | .dd => .div_dd
  | .ofAnd h => match h with
    | .ff => .and_ff | .fn => .and_fn | .fd => .and_fd
    | .nf => .and_nf | .nn => .and_nn | .nd => .and_nd
    | .df => .and_df | .dn => .and_dn | .dd => .and_dd
  | .ofOr  h => match h with
    | .ff => .or_ff  | .fn => .or_fn  | .fd => .or_fd
    | .nf => .or_nf  | .nn => .or_nn  | .nd => .or_nd
    | .df => .or_df  | .dn => .or_dn  | .dd => .or_dd
  | .ofEq  h => match h with
    | .ff => .eq_ff  | .fn => .eq_fn  | .fd => .eq_fd
    | .nf => .eq_nf  | .nn => .eq_nn  | .nd => .eq_nd
    | .df => .eq_df  | .dn => .eq_dn  | .dd => .eq_dd
  | .ofNe  h => match h with
    | .ff => .ne_ff  | .fn => .ne_fn  | .fd => .ne_fd
    | .nf => .ne_nf  | .nn => .ne_nn  | .nd => .ne_nd
    | .df => .ne_df  | .dn => .ne_dn  | .dd => .ne_dd
  | .ofLt  h => match h with
    | .ff => .lt_ff  | .fn => .lt_fn  | .fd => .lt_fd
    | .nf => .lt_nf  | .nn => .lt_nn  | .nd => .lt_nd
    | .df => .lt_df  | .dn => .lt_dn  | .dd => .lt_dd
  | .ofLe  h => match h with
    | .ff => .le_ff  | .fn => .le_fn  | .fd => .le_fd
    | .nf => .le_nf  | .nn => .le_nn  | .nd => .le_nd
    | .df => .le_df  | .dn => .le_dn  | .dd => .le_dd

/-! ### Compile typeclass instances -/


/-- Random contexts compile to tensor shapes. -/
@[reducible] instance : Util.Compile RandomContext where
  Target := Shape
  compile := (↑·)

/-- Well-formed random contexts compile to tensor shapes. -/
@[reducible] instance : Util.Compile WellFormedRandomContext where
  Target := Shape
  compile Δ := (↑Δ : Shape)

/-- Compile a static context to a JAX context. Each static parameter `⟨x, T⟩`
    becomes a scalar tensor `⟨x, JAX.Ty.tensor [] (Ty.compile T)⟩`. -/
def StaticContext.toJAXContext : StaticContext → JAX.Context
  | [] => []
  | ⟨x, T⟩ :: rest => ⟨x, JAX.Ty.tensor [] 𝒞⟦T⟧⟩ :: StaticContext.toJAXContext rest

/-- Static contexts compile to JAX contexts (scalar tensor per parameter). -/
@[reducible] instance : Util.Compile StaticContext where
  Target := JAX.Context
  compile := StaticContext.toJAXContext

@[simp] theorem StaticContext.toJAXContext_keys (Θ : StaticContext) :
    (StaticContext.toJAXContext Θ).keys = Θ.keys := by
  induction Θ with
  | nil => rfl
  | cons p rest ih => cases p; simp only [toJAXContext, List.keys_cons, ih]

theorem StaticContext.toJAXContext_nodupKeys {Θ : StaticContext}
    (h : Θ.NodupKeys) : (StaticContext.toJAXContext Θ).NodupKeys := by
  rw [List.NodupKeys, toJAXContext_keys]; exact h

theorem StaticContext.toJAXContext_mem {x : String} {T : PAL.Ty}
    {Θ : StaticContext}
    (h : ⟨x, T⟩ ∈ Θ) :
    ⟨x, JAX.Ty.tensor [] 𝒞⟦T⟧⟩ ∈ StaticContext.toJAXContext Θ := by
  induction h with
  | head => exact .head _
  | tail _ _ ih => exact .tail _ ih

/-- Static parameter membership lifts to a suffix-appended context. -/
theorem StaticContext.toJAXContext_mem_append {x : String} {T : PAL.Ty}
    {Θ : StaticContext} (Γ : JAX.Context)
    (h : ⟨x, T⟩ ∈ Θ) :
    ⟨x, JAX.Ty.tensor [] 𝒞⟦T⟧⟩ ∈ Γ ++ StaticContext.toJAXContext Θ :=
  List.mem_append.mpr (Or.inr (StaticContext.toJAXContext_mem h))

/-! ### Compilation Context Instances -/

open Palladium.PAL in
/-- Well-formed static contexts compile to well-formed JAX contexts. -/
@[reducible] instance : Util.Compile WellFormedStaticContext where
  Target := JAX.WellFormedContext
  compile Θ := ⟨𝒞⟦(↑Θ : StaticContext)⟧, StaticContext.toJAXContext_nodupKeys Θ.wf⟩

open Palladium.PAL in
/-- Valid static contexts compile to JAX contexts. -/
@[reducible] instance : Util.Compile ValidStaticContext where
  Target := JAX.Context
  compile Θ := 𝒞⟦(↑Θ : StaticContext)⟧

/-- The reserved `probabilityTensor` name never appears among compiled static-parameter keys. -/
theorem ValidStaticContext.probabilityTensor_not_in_jax_keys {Θ : ValidStaticContext} :
    probabilityTensor ∉ (𝒞⟦Θ⟧ : JAX.Context).keys := by
  intro h
  have hkeys : (𝒞⟦Θ⟧ : JAX.Context).keys = (↑Θ : StaticContext).keys :=
    StaticContext.toJAXContext_keys (↑Θ : StaticContext)
  rw [hkeys] at h
  obtain ⟨T, hmem⟩ := List.mem_keys.mp h
  have hpr := (Θ.paramsNotReserved : NameInvariant Θ.ctx) probabilityTensor hmem
  simp [isReserved_probabilityTensor] at hpr

/-- If `probabilityTensor` is absent from `Γ` and from `𝒞⟦Θ⟧`, it is absent from their append. -/
theorem probabilityTensor_not_in_append {Γ : JAX.Context} {Θ : ValidStaticContext}
    (hΓ : probabilityTensor ∉ Γ.keys) :
    probabilityTensor ∉ (Γ ++ 𝒞⟦Θ⟧).keys := by
  intro h
  rw [List.keys_append] at h
  cases List.mem_append.mp h with
  | inl h' => exact hΓ h'
  | inr h' => exact ValidStaticContext.probabilityTensor_not_in_jax_keys h'

end Palladium.Compiler
