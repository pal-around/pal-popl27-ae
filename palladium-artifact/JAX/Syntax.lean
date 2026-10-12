import JAX.Operators
import Util.Context
import Util.Denote
import Util.Discrete
import TensorLib.Notations

namespace Palladium.JAX

open Palladium.Util
open Palladium.TensorLib
open Shape

/-- JAX types with explicit array annotations. -/
inductive Ty : Type where
  /-- Tensor with shape annotation and element type. -/
  | tensor : (σ : Shape) -> (η : DType) -> Ty
deriving Repr, BEq, Inhabited, DecidableEq

/-- A typing context maps variable names to types. -/
abbrev Context := Util.Context Ty

/-- A well-formed typing context with unique keys. -/
abbrev WellFormedContext := Util.WellFormedContext Ty

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-!
JAX typing rules (Unicode).

We write `Γ ⊢ e : τ` for JAX expressions and `Γ ⊢ s ⊣ Γ'` for statements.
All JAX types are arrays `array σ dtype`; scalars are rank-0 arrays with
shape `[]`. Unary/binary ops coerce numeric results to `float` and
boolean/relational results to `disc(.fin 2)`. Broadcasting uses `⋈`
(`Shape.BroadcastCompatible`) with compile-time proofs.
-/

/-- Typed JAX expressions indexed by a layout `L`, context, and type. -/
inductive Expr : Context -> Ty -> Type
  --  Γ(x) = τ
  --  ───────────────  (VAR)
  --  Γ ⊢ x : τ
  | var : (name : String) -> (T : Ty) ->
      (h : ⟨name, T⟩ ∈ Γ) ->
      Expr Γ T
  --  t : Tensor σ ⟦η⟧
  --  ───────────────────────────────────────  (ARRAY)
  --  Γ ⊢ array(t) : tensor σ η
  | array : {σ : Shape} -> (η : DType) ->
      (t : Tensor (L := L) σ ⟦η⟧) ->
      Expr Γ (Ty.tensor σ η)
  --  Γ ⊢ e : tensor σ η    UOp.Valid op η ηₒ
  --  ────────────────────────────────────────────────  (UOP)
  --  Γ ⊢ uop e : tensor σ ηₒ
  | uop : (op : UOp) -> {σ : Shape} -> {η ηₒ : DType} ->
      Expr Γ (Ty.tensor σ η) -> UOp.Valid op η ηₒ ->
      Expr Γ (Ty.tensor σ ηₒ)
  --  Γ ⊢ e₁ : tensor σ₁ η₁    Γ ⊢ e₂ : tensor σ₂ η₂
  --  BOp.Valid op η₁ η₂ ηₒ    BroadcastCompatible σ₁ σ₂
  --  ──────────────────────────────────────────────────────────────  (BOP)
  --  Γ ⊢ e₁ bop e₂ : tensor (σ₁ ⊔ σ₂ ‖ h) ηₒ
  | bop : (op : BOp) ->
      {σ₁ σ₂ : Shape} -> {η₁ η₂ ηₒ : DType} ->
      Expr Γ (Ty.tensor σ₁ η₁) -> Expr Γ (Ty.tensor σ₂ η₂) ->
      BOp.Valid op η₁ η₂ ηₒ ->
      (h : σ₁ ⋈ σ₂) ->
      Expr Γ (Ty.tensor (σ₁ ⊔ σ₂ ‖ h) ηₒ)
  --  Γ ⊢ cond : tensor σ₃ bool    Γ ⊢ x : tensor σ₁ η₁    Γ ⊢ y : tensor σ₂ η₂
  --  BroadcastNCompatible #v[σ₃, σ₁, σ₂]
  --  ──────────────────────────────────────────────────────────────────────  (SELECT)
  --  Γ ⊢ select(cond, x, y) : tensor (⊔ₙ #v[σ₃, σ₁, σ₂] ‖ h) (η₁ ⊔ η₂)
  | select : {σ₃ σ₁ σ₂ : Shape} -> {η₁ η₂ : DType} ->
      (cond : Expr Γ (Ty.tensor σ₃ DType.bool)) ->
      (x : Expr Γ (Ty.tensor σ₁ η₁)) ->
      (y : Expr Γ (Ty.tensor σ₂ η₂)) ->
      (h : ⋈ₙ #v[σ₃, σ₁, σ₂]) ->
      Expr Γ (Ty.tensor (⊔ₙ #v[σ₃, σ₁, σ₂] ‖ h) (η₁ ⊔ η₂))
  --  Γ ⊢ e : tensor σ η
  --  ────────────────────────────────────  (ONES-LIKE)
  --  Γ ⊢ ones_like(e) : tensor σ float
  | onesLike : {σ : Shape} -> {η : DType} ->
      Expr Γ (Ty.tensor σ η) -> Expr Γ (Ty.tensor σ .float)
  --  s : Shape
  --  ────────────────────────────────  (ONES)
  --  Γ ⊢ ones(s) : tensor s float
  | ones : (σ : Shape) -> Expr Γ (Ty.tensor σ .float)
  --  Γ ⊢ e : tensor σ η    σ ⊔ σₒ = σₒ
  --  ─────────────────────────────────────────────  (BROADCAST-TO)
  --  Γ ⊢ broadcast_to(e, σₒ) : tensor σₒ η
  | broadcastTo : {σ : Shape} -> {η : DType} ->
      Expr Γ (Ty.tensor σ η) -> (σₒ : Shape) ->
      (h : Broadcastable σ σₒ) ->
      Expr Γ (Ty.tensor σₒ η)
  --  Γ ⊢ arr : tensor σₛ η    Reshapeable σₛ σₜ
  --  ─────────────────────────────────────────────  (RESHAPE)
  --  Γ ⊢ reshape(arr, σₜ) : tensor σₜ η
  | reshape : {σₛ : Shape} -> {η : DType} ->
      Expr Γ (Ty.tensor σₛ η) -> (σₜ : Shape) ->
      (h : σₛ ≅ₛ σₜ) ->
      Expr Γ (Ty.tensor σₜ η)
  --  Γ ⊢ e : tensor σ η    axes nodup
  --  ────────────────────────────────────────────────────────────────────────────  (SUM-AXES)
  --  Γ ⊢ sum(e, axes) : tensor (removeAxes σ axes) (η.sumOutMany axes.length)
  | sum : {σ : Shape} -> {η : DType} ->
          Expr Γ (Ty.tensor σ η) -> (axes : List (Fin σ.rank)) ->
          (h : axes.Nodup) ->
          Expr Γ (Ty.tensor (removeAxes σ axes) (η.sumOutMany axes.length))
  --  Γ ⊢ e : tensor σ η    Squeezable σ axis
  --  ──────────────────────────────────────────────  (SQUEEZE-AXIS)
  --  Γ ⊢ squeeze(e, axis) : tensor (σ.slice axis) η
  | squeeze : {σ : Shape} -> {η : DType} ->
              Expr Γ (Ty.tensor σ η) -> (axis : Fin σ.rank) ->
              (h : Shape.Squeezable σ axis) ->
              Expr Γ (Ty.tensor (σ.slice axis) η)
  --  Γ ⊢ e : tensor σ η    Permutation.Valid perm σ.rank
  --  ─────────────────────────────────────────────────────────  (TRANSPOSE)
  --  Γ ⊢ transpose(e, perm) : tensor (σ.permute perm) η
  | transpose : {σ : Shape} -> {η : DType} ->
                Expr Γ (Ty.tensor σ η) -> (perm : Permutation) ->
                (h : Permutation.Valid perm σ.rank) ->
                Expr Γ (Ty.tensor (σ.permute perm h) η)
  --  Γ ⊢ e₁ : τ₁    ⟨name, τ₁⟩ :: Γ ⊢ e₂ : τ₂
  --  ────────────────────────────────────────────────  (LET-IN)
  --  Γ ⊢ let name = e₁ in e₂ : τ₂
  | letIn : (name : String) -> {T₁ T₂ : Ty} ->
            Expr Γ T₁ -> Expr (⟨name, T₁⟩ :: Γ) T₂ ->
            Expr Γ T₂
  --  Γ ⊢ e : τ
  --  ──────────────────────────────  (COMMENT)
  --  Γ ⊢ comment "..." e : τ
  /-- A no-op wrapper that carries a comment string. -/
  | comment : String -> Expr Γ T -> Expr Γ T
  --  Γ ⊢ e : τ    Γ, x : τ ⊢ e' : τ
  --  ─────────────────────────────────  (SCAN)
  --  Γ ⊢ scan x ← e for n in e' : τ
  /-- Bounded loop with loop-carried state. Iterates `body` `n` times
      starting from `init`. The body is typed in the extended context
      with the loop variable bound. -/
  | scan : (name : String) -> {T : Ty} ->
      (init : Expr Γ T) ->
      (n : Nat) ->
      (body : Expr (⟨name, T⟩ :: Γ) T) ->
      Expr Γ T

/-- Whether a JAX expression uses `letIn` (PAL.Expr.compile never emits `letIn`). -/
def Expr.hasLetIn {Γ : Context} {τ : Ty} : Expr (L := L) Γ τ → Bool
  | .var .. => false
  | .array .. => false
  | .uop _ e .. => hasLetIn e
  | .bop _ e₁ e₂ .. => e₁.hasLetIn || e₂.hasLetIn
  | .select e₁ e₂ e₃ .. => e₁.hasLetIn || e₂.hasLetIn || e₃.hasLetIn
  | .onesLike e .. => hasLetIn e
  | .ones .. => false
  | .broadcastTo e .. => hasLetIn e
  | .reshape e .. => hasLetIn e
  | .sum e .. => hasLetIn e
  | .squeeze e .. => hasLetIn e
  | .transpose e .. => hasLetIn e
  | .letIn .. => true
  | .comment _ e .. => hasLetIn e
  | .scan .. => true  -- scan has binding structure like letIn; not supported in weakenMask

end

section

variable {L : Tensor.Layout}

/-- Casting along an element-type (`DType`) equality does not change `hasLetIn`. -/
@[simp] theorem Expr.hasLetIn_cast_dtype {Γ : Context} {σ : Shape} {η η' : DType} (h : η = η')
    (e : Expr (L := L) Γ (Ty.tensor σ η)) :
    (h ▸ e : Expr (L := L) Γ (Ty.tensor σ η')).hasLetIn = e.hasLetIn := by
  subst h
  rfl

end

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-- Look up a variable in context, returning the appropriately typed expression.
    Used internally by the `jax{...}` macro to resolve identifiers. -/
def Expr.ref (Γ : Context) (name : String) :
    Expr (L := L) Γ
      (match List.dlookup name Γ with
      | .some T => T
      | .none => Ty.tensor [] DType.bool) :=
  match h : List.dlookup name Γ with
  | .some T => Expr.var name T (List.of_mem_dlookup h)
  | .none =>
      Expr.array (σ := []) DType.bool
        ((panic! s!"unbound variable {name}") : Tensor (L := L) [] (Fin 2))

/-! ### Unary operator helpers -/

/-- Numeric negation (`-e`). -/
@[reducible] def Expr.neg (e : Expr (L := L) Γ (Ty.tensor σ η)) :
    Expr (L := L) Γ (Ty.tensor σ .float) :=
  .uop .neg e .neg

/-- Boolean negation. -/
@[reducible] def Expr.not (e : Expr (L := L) Γ (Ty.tensor σ η)) :
    Expr (L := L) Γ (Ty.tensor σ DType.bool) :=
  .uop .not e .not

/-! ### Binary operator helpers (macro-facing, take `BroadcastCompatible` proof) -/

/-- Addition with broadcasting. -/
@[reducible] def Expr.add (a : Expr (L := L) Γ (Ty.tensor σ₁ η₁))
    (b : Expr (L := L) Γ (Ty.tensor σ₂ η₂))
    (hc : σ₁ ⋈ σ₂) :=
  Expr.bop .add a b .add hc

/-- Subtraction with broadcasting. -/
@[reducible] def Expr.sub (a : Expr (L := L) Γ (Ty.tensor σ₁ η₁))
    (b : Expr (L := L) Γ (Ty.tensor σ₂ η₂))
    (hc : σ₁ ⋈ σ₂) :=
  Expr.bop .sub a b .sub hc

/-- Multiplication with broadcasting. -/
@[reducible] def Expr.mul (a : Expr (L := L) Γ (Ty.tensor σ₁ η₁))
    (b : Expr (L := L) Γ (Ty.tensor σ₂ η₂))
    (hc : σ₁ ⋈ σ₂) :=
  Expr.bop .mul a b .mul hc

/-- Division with broadcasting. -/
@[reducible] def Expr.div (a : Expr (L := L) Γ (Ty.tensor σ₁ η₁))
    (b : Expr (L := L) Γ (Ty.tensor σ₂ η₂))
    (hc : σ₁ ⋈ σ₂) :=
  Expr.bop .div a b .div hc

/-- Boolean conjunction with broadcasting. -/
@[reducible] def Expr.and (a : Expr (L := L) Γ (Ty.tensor σ₁ η₁))
    (b : Expr (L := L) Γ (Ty.tensor σ₂ η₂))
    (hc : σ₁ ⋈ σ₂) :=
  Expr.bop .and a b .and hc

/-- Boolean disjunction with broadcasting. -/
@[reducible] def Expr.or (a : Expr (L := L) Γ (Ty.tensor σ₁ η₁))
    (b : Expr (L := L) Γ (Ty.tensor σ₂ η₂))
    (hc : σ₁ ⋈ σ₂) :=
  Expr.bop .or a b .or hc

/-- Equality comparison with broadcasting. -/
@[reducible] def Expr.eq (a : Expr (L := L) Γ (Ty.tensor σ₁ η₁))
    (b : Expr (L := L) Γ (Ty.tensor σ₂ η₂))
    (hc : σ₁ ⋈ σ₂) :=
  Expr.bop .eq a b .eq hc

/-- Inequality comparison with broadcasting. -/
@[reducible] def Expr.ne (a : Expr (L := L) Γ (Ty.tensor σ₁ η₁))
    (b : Expr (L := L) Γ (Ty.tensor σ₂ η₂))
    (hc : σ₁ ⋈ σ₂) :=
  Expr.bop .ne a b .ne hc

/-- Less-than comparison with broadcasting. -/
@[reducible] def Expr.lt (a : Expr (L := L) Γ (Ty.tensor σ₁ η₁))
    (b : Expr (L := L) Γ (Ty.tensor σ₂ η₂))
    (hc : σ₁ ⋈ σ₂) :=
  Expr.bop .lt a b .lt hc

/-- Less-or-equal comparison with broadcasting. -/
@[reducible] def Expr.le (a : Expr (L := L) Γ (Ty.tensor σ₁ η₁))
    (b : Expr (L := L) Γ (Ty.tensor σ₂ η₂))
    (hc : σ₁ ⋈ σ₂) :=
  Expr.bop .le a b .le hc

/-- Greater-than comparison with broadcasting. -/
@[reducible] def Expr.gt (a : Expr (L := L) Γ (Ty.tensor σ₁ η₁))
    (b : Expr (L := L) Γ (Ty.tensor σ₂ η₂))
    (hc : σ₁ ⋈ σ₂) :=
  Expr.bop .gt a b .gt hc

/-- Greater-or-equal comparison with broadcasting. -/
@[reducible] def Expr.ge (a : Expr (L := L) Γ (Ty.tensor σ₁ η₁))
    (b : Expr (L := L) Γ (Ty.tensor σ₂ η₂))
    (hc : σ₁ ⋈ σ₂) :=
  Expr.bop .ge a b .ge hc

/-! ### Select helper -/

/-- Conditional select with broadcasting (macro helper).
    Takes a single `BroadcastNCompatible` proof and passes it through to `select`.
    Used by the `jax{ select(...) }` macro. -/
@[reducible] def Expr.mkSelect
    (cond : Expr (L := L) Γ (Ty.tensor σ₃ DType.bool))
    (x : Expr (L := L) Γ (Ty.tensor σ₁ η₁))
    (y : Expr (L := L) Γ (Ty.tensor σ₂ η₂))
    (h : ⋈ₙ #v[σ₃, σ₁, σ₂]) :
    Expr (L := L) Γ (Ty.tensor (⊔ₙ #v[σ₃, σ₁, σ₂] ‖ h) (η₁ ⊔ η₂)) :=
  .select cond x y h

/- JAX statements, indexed by input and output typing contexts. -/
inductive Stmt (Γ : Context) : Context -> Type where
  --  Γ ⊢ e : τ
  --  ─────────────────────────────  (ASSIGN)
  --  Γ ⊢ x := e ⊣ Γ[x ↦ τ]
  | assign : (name : String) -> Expr (L := L) Γ T -> Stmt Γ (List.kinsert name T Γ)
  --  ──────────────────────────────  (COMMENT)
  --  Γ ⊢ comment "..." ⊣ Γ
  | comment : String -> Stmt Γ Γ
  --  ─────────────────────────────  (DELETE)
  --  Γ ⊢ delete x ⊣ Γ \ x
  | delete : (name : String) -> Stmt Γ (List.kerase name Γ)

/- Sequential statement blocks, threading contexts left-to-right. -/
inductive Stmts : Context -> Context -> Type where
  --  ───────────────────  (NIL)
  --  Γ ⊢ [] ⊣ Γ
  | nil : Stmts Γ Γ
  --  Γ ⊢ s ⊣ Γ'    Γ' ⊢ ss ⊣ Γ''
  --  ──────────────────────────────────  (CONS)
  --  Γ ⊢ s; ss ⊣ Γ''
  | cons : Stmt (L := L) Γ Γ' -> Stmts Γ' Γ'' -> Stmts Γ Γ''

def Stmts.append : Stmts (L := L) Γ Γ' -> Stmts (L := L) Γ' Γ'' -> Stmts (L := L) Γ Γ''
  | Stmts.nil, ys => ys
  | Stmts.cons x xs, ys => Stmts.cons x (Stmts.append xs ys)

instance : HAppend (Stmts (L := L) Γ Γ') (Stmts (L := L) Γ' Γ'') (Stmts (L := L) Γ Γ'') where
  hAppend := Stmts.append



--  Γ ⊢ body ⊣ Γ'    Γ'(ret) = retTy
--  ─────────────────────────────────────────────────────  (PROGRAM)
--  program(name, Γ, body, ret) : Γ → retTy

/-- JAX function declaration with parameters, body, and return variable. -/
inductive Program (argTy : Context) (retTy : Ty) : Type where
  /-- Construct a declaration with a name, body, return variable, and proof that it exists. -/
  | mk
      (name : String)
      (body : Stmts (L := L) argTy Γ')
      (ret : String)
      (h : ⟨ret, retTy⟩ ∈ Γ') :
      Program argTy retTy

end

end Palladium.JAX
