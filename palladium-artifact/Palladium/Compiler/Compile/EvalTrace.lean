import Palladium.Compiler.Compile.Marginalize
import Palladium.Compiler.Compile.Sample

namespace Palladium.PAL.Stmt

open Palladium.Util
open Palladium.PAL
open Palladium.Compiler
open Palladium.TensorLib

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-! ### Eval Trace Helpers -/

/-- Compute `randomAxis` from a known index position. -/
theorem randomAxis_eq_of_getElem
    (Δ : RandomContext) (name : String) (D : Discrete)
    (h : ⟨name, D⟩ ∈ Δ) (idx : Nat)
    (hIdxLt : idx < Δ.keys.length)
    (hKeysIdx : Δ.keys[idx]'hIdxLt = name)
    (hnodup : Δ.keys.Nodup) :
    randomAxis Δ name D h = idx := by
  have h_some : Δ.keys.findIdx? (· == name) = some idx :=
    List.findIdx?_eq_some_iff_getElem.mpr
      ⟨hIdxLt, by rw [hKeysIdx]; exact beq_self_eq_true name,
        fun j hj hjlt => by
          have : Δ.keys[j]'(by omega) ≠ Δ.keys[idx]'hIdxLt :=
            List.pairwise_iff_getElem.mp hnodup j idx (by omega) hIdxLt hj
          rw [hKeysIdx] at this; exact this (beq_iff_eq.mp hjlt)⟩
  have h_ra := randomAxis_eq_findIdx? Δ name D h
  rw [h_some] at h_ra; exact Option.some_injective _ h_ra.symm


end

end Palladium.PAL.Stmt
