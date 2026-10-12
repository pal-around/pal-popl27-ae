import Palladium.Compiler.Compile.Sample
import Palladium.Compiler.Compile.Marginalize
import Palladium.Compiler.Compile.Condition
import Palladium.Compiler.Compile.ReturnPerm

/-! ### Statement Dispatch -/

namespace Palladium.PAL

open Palladium.Util
open Palladium.Compiler
open Palladium.TensorLib
open Shape
open Stmt

variable {L : Tensor.Layout} [Tensor.Operations L]
  [Tensor.OfTabulated L]

/-- Right-fold `c` same-shape float tensor expressions with `bop .add`, seeded with
    an all-zeros tensor literal of shape `σ`. Used to assemble the stochastic-`repeat`
    mixture (`Σ_{k<c} termₖ`) over a static bound `c`. The fold is zero-seeded so that
    the `c = 0`/`c = 1` cases are uniform; the seed indexes to `0.0` everywhere
    (`Tensor.index_fill`). -/
def Stmt.mixtureAdd {Γ : JAX.Context} (σ : Shape) (c : Nat)
    (term : Fin c → JAX.Expr (L := L) Γ (JAX.Ty.tensor σ .float)) :
    JAX.Expr (L := L) Γ (JAX.Ty.tensor σ .float) :=
  (List.finRange c).foldr
    (fun k acc =>
      have hbc : Shape.BroadcastCompatible σ σ :=
        (Shape.Broadcastable.refl σ).toBroadcastCompatible
      have hshape : Shape.broadcast2 σ σ hbc = σ :=
        (Shape.Broadcastable.refl σ).broadcast2_eq_right hbc
      hshape ▸ JAX.Expr.bop .add (term k) acc .add_ff hbc)
    (JAX.Expr.array .float (Tensor.fill (L := L) (0.0 : Float) σ))

mutual
/-- Compile a PAL statement (including sequencing) to a JAX expression.
    Takes the current probability tensor expression and returns the updated one.
    Rules: C-Skip (`nil`), C-Seq (`seq`), C-Sample (`sample`), C-Marg (`marginalize`),
    C-Condition (`condition`), C-If (`ite`); `repeat` compiles each iteration count with
    the `scan` of rule C-Repeat. -/
def Stmt.compile {Δ Δ' : WellFormedRandomContext}
    {Γ : JAX.Context}
    (s : Stmt (Θ := Θ) Δ Δ')
    (ptExpr : JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ⟧ .float)) :
    JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ'⟧ .float) :=
  match s with
  | Stmt.nil _ => ptExpr
  | Stmt.seq first rest =>
      Stmt.compile rest (Stmt.compile first ptExpr)
  | Stmt.sample vars hwf wppc =>
      compileSample vars hwf wppc ptExpr
  | Stmt.marginalize name D hmem =>
      compileMarginalize name D hmem ptExpr
  | Stmt.condition c =>
      compileCondition c ptExpr
  | Stmt.ite (Δ' := carry) lhs cond then_block else_block htypes_then htypes_else hwf =>
      let outCtx : WellFormedRandomContext := ⟨↑lhs ++ ↑carry, hwf⟩
      let maskedTrue := compileCondition (L := L) cond ptExpr
      let maskedFalse := compileCondition (L := L) (.not cond) ptExpr
      let r1' := Block.compileAs then_block htypes_then hwf maskedTrue
      let r2' := Block.compileAs else_block htypes_else hwf maskedFalse
      have hbc := (Broadcastable.refl 𝒞⟦outCtx⟧).toBroadcastCompatible
      have hshape := (Broadcastable.refl 𝒞⟦outCtx⟧).broadcast2_eq_right hbc
      hshape ▸ JAX.Expr.bop .add r1' r2' .add_ff hbc
  | Stmt.repeat (Δ_loop_start := Δ_loop_start) (c := c) (hc := hc)
      lhs hwf n initBlock loopBlock htypes_init htypes_body =>
      let loopCtx : WellFormedRandomContext := ⟨↑lhs ++ ↑Δ_loop_start, hwf⟩
      let zName := probabilityTensor ++ "_scan"
      let zType := JAX.Ty.tensor 𝒞⟦loopCtx⟧ .float
      let zBind : Σ _ : String, JAX.Ty := ⟨zName, zType⟩
      let zVar : JAX.Expr (L := L) ((zBind :: Γ) ++ 𝒞⟦Θ⟧) zType :=
        JAX.Expr.var zName zType (.head _)
      let bodyExpr : JAX.Expr (L := L) ((zBind :: Γ) ++ 𝒞⟦Θ⟧)
          (JAX.Ty.tensor 𝒞⟦loopCtx⟧ .float) :=
        Block.compileAs loopBlock htypes_body hwf zVar
      -- The init block is compiled once as a function of its (masked) input
      -- probability tensor; this keeps the recursive call at the top level.
      let compileInit :
          JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ⟧ .float) →
          JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦loopCtx⟧ .float) :=
        Block.compileAs initBlock htypes_init hwf
      -- Stochastic loop: mixture over k = 0..c-1 of "run the loop `k` times on the
      -- slice of the input where `n = k`". Masking is `condition (n == k)`; unrolling
      -- `k` times is a static `scan`.
      Stmt.mixtureAdd 𝒞⟦loopCtx⟧ c (fun k =>
        JAX.Expr.scan zName
          (compileInit (compileCondition (L := L)
            (Expr.eq n (Expr.fin (hn := hc) k)) ptExpr))
          k.val bodyExpr)

/-- Compile a block and cast its output shape to `lhs ++ Δ'` (the caller's
    output context). Used by `Stmt.compile` ite to align both branches'
    outputs. The trailing shape cast `𝒞⟦Δ_ret ++ Δ'⟧ = 𝒞⟦lhs ++ Δ'⟧`
    is built from the shared
    `Block.htypes_full htypes` so `Stmt.correctness` can quote a single
    proof term — avoiding proof-irrelevance mismatches under `simp`. -/
def Block.compileAs {Δ Δ' Δ_ret lhs : WellFormedRandomContext} {Γ : JAX.Context}
    (blk : Block (Θ := Θ) Δ Δ' Δ_ret)
    (htypes : (↑lhs : RandomContext).map Sigma.snd =
        (↑Δ_ret : RandomContext).map Sigma.snd)
    (hwf : (↑lhs ++ ↑Δ' : RandomContext).NodupKeys)
    (ptExpr : JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ⟧ .float)) :
    JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧)
      (JAX.Ty.tensor 𝒞⟦(⟨↑lhs ++ ↑Δ', hwf⟩ : WellFormedRandomContext)⟧ .float) :=
  match blk with
  | Block.mk (Δ_ret_var := Δ_ret_var) hperm hwf_ext body =>
      let bodyCtx : WellFormedRandomContext :=
        ⟨↑Δ_ret_var ++ ↑Δ', blockInternalNodup hperm hwf_ext⟩
      let extCtx : WellFormedRandomContext := ⟨↑Δ_ret ++ ↑Δ', hwf_ext⟩
      let extCtx_lhs : WellFormedRandomContext := ⟨↑lhs ++ ↑Δ', hwf⟩
      let bodyExpr := Stmt.compile body ptExpr
      let hret_full : (↑extCtx : RandomContext).Perm ↑bodyCtx :=
        List.Perm.append_right _ hperm.symm
      let returnPerm := Block.computeReturnPerm ↑bodyCtx extCtx.keys
      have hperm_keys : extCtx.keys.Perm bodyCtx.keys := hret_full.map Sigma.fst
      have hPermValid : Permutation.Valid returnPerm 𝒞⟦bodyCtx⟧.rank := by
        rw [bodyCtx.toShape_length]
        exact Block.computeReturnPerm_valid ↑bodyCtx extCtx.keys bodyCtx.wf hperm_keys
      have hshape_perm : 𝒞⟦bodyCtx⟧.permute returnPerm hPermValid = 𝒞⟦extCtx⟧ :=
        Block.toShape_permute_computeReturnPerm ↑bodyCtx (↑extCtx)
          hret_full extCtx.keys_nodup hPermValid
      have hshape_lhs : (𝒞⟦extCtx⟧ : Shape) = 𝒞⟦extCtx_lhs⟧ :=
        RandomContext.toShape_eq_of_map_snd_eq (Block.htypes_full htypes).symm
      if hctx : (↑Δ_ret_var : RandomContext) = (↑Δ_ret : RandomContext) then
        have hshape_body_ext : 𝒞⟦bodyCtx⟧ = 𝒞⟦extCtx⟧ := by
          change RandomContext.toShape (↑bodyCtx : RandomContext) =
            RandomContext.toShape (↑extCtx : RandomContext)
          simp only [bodyCtx, extCtx]
          rw [hctx]
        (hshape_body_ext.trans hshape_lhs) ▸ bodyExpr
      else
        (hshape_perm.trans hshape_lhs) ▸ JAX.Expr.transpose bodyExpr returnPerm hPermValid
end

/-- Compile a PAL block at its declared return context (rule C-Block). -/
def Block.compile {Δ Δ' Δ_ret : WellFormedRandomContext} {Γ : JAX.Context}
    (blk : Block (Θ := Θ) Δ Δ' Δ_ret)
    (ptExpr : JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ⟧ .float)) :
    JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧)
      (JAX.Ty.tensor 𝒞⟦blk.outputContext⟧ .float) :=
  match blk with
  | Block.mk _ hwf_ext _ =>
      Block.compileAs blk rfl hwf_ext ptExpr

end Palladium.PAL

namespace Palladium.PAL.Stmt

open Palladium.Util Palladium.Compiler Palladium.TensorLib

/-- PAL statements compile to JAX expression transformers. -/
@[reducible] instance {Θ : ValidStaticContext} {Δ Δ' : WellFormedRandomContext} :
    Util.Compile (Stmt (Θ := Θ) Δ Δ') where
  Target := ∀ {L : Tensor.Layout} [Tensor.Operations L] [Tensor.OfTabulated L] {Γ : JAX.Context},
    JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ⟧ .float) →
    JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ'⟧ .float)
  compile s := fun {_} [_] [_] {Γ} e =>
    e |> s.compile (Γ := Γ)

end Palladium.PAL.Stmt
