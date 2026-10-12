import TensorLib.Shape.Basics

namespace Palladium.TensorLib
namespace Shape

/-! ## Reshapeable -/

/-- `Reshapeable σₛ σₜ` means `σₛ` can be reshaped to `σₜ` (same total element count). -/
def Reshapeable (σₛ σₜ : Shape) : Prop := volume σₛ = volume σₜ

instance : Decidable (Reshapeable σₛ σₜ) :=
  if h : volume σₛ = volume σₜ then
    .isTrue h
  else
    .isFalse h


/-- `[n]` is reshapeable to `n :: List.replicate k 1`. -/
theorem Reshapeable.enum (n k : Nat) : Reshapeable [n] (n :: List.replicate k 1) := by
  simp [Reshapeable, volume_cons, volume_replicate_one, volume_nil]

end Shape
end Palladium.TensorLib
