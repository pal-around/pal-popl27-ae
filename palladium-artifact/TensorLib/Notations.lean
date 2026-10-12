import TensorLib.DType
import TensorLib.Shape.Broadcastable
import TensorLib.Shape.Reshapeable
import TensorLib.Tensor

/-!
# TensorLib Notations

Centralized notation declarations for TensorLib. Downstream consumers (PAL, JAX,
Compiler, Testing) import this file to get all TensorLib notations. TensorLib
internal files must use explicit forms to avoid import cycles.
-/

namespace Palladium.TensorLib

/-- Notation for dtype promotion (join/supremum in the type lattice). -/
scoped infixl:68 " ⊔ " => DType.promote

namespace Shape

/-- Broadcast compatibility notation: `σ₁ ⋈ σ₂` iff `BroadcastCompatible σ₁ σ₂`. -/
scoped notation:50 lhs " ⋈ " rhs => BroadcastCompatible lhs rhs

/-- `lhs ⊔ rhs ‖ h` is the broadcast join of shapes `lhs` and `rhs` with compatibility proof `h`. -/
scoped notation:65 lhs:max " ⊔ " rhs:max " ‖ " h:65 => broadcast2 lhs rhs h

/-- Reshapeable notation: `σₛ ≅ₛ σₜ` iff `Reshapeable σₛ σₜ`. -/
scoped notation:50 lhs " ≅ₛ " rhs => Reshapeable lhs rhs

/-- `⋈ₙ shapes` is N-way broadcast compatibility of `shapes`. -/
scoped notation:50 "⋈ₙ " shapes => BroadcastNCompatible shapes

/-- `⊔ₙ shapes ‖ h` is the N-way broadcast join of `shapes` with compatibility proof `h`. -/
scoped notation:65 "⊔ₙ " shapes:max " ‖ " h:65 => broadcastN shapes h

/-- Broadcastability order on shapes: `a ≤ b` iff `a` can be `Broadcastable` to `b`. -/
scoped instance : LE Shape where
  le := Broadcastable

/-- Strict broadcastability: `a < b` iff `a` broadcasts to `b` but not vice versa. -/
scoped instance : LT Shape where
  lt a b := a ≤ b ∧ ¬(b ≤ a)

/-- Broadcastability partial order on shapes. -/
scoped instance : PartialOrder Shape where
  le_refl := Broadcastable.refl
  le_trans _ _ _ := Broadcastable.trans
  lt_iff_le_not_ge := by intros; rfl
  le_antisymm _ _ := Broadcastable.antisymm

scoped instance (σ₁ σ₂ : Shape) : Decidable (σ₁ ≤ σ₂) :=
  inferInstanceAs (Decidable (Broadcastable σ₁ σ₂))

end Shape

namespace Tensor

/-- `t ▷ target ‖ h` broadcasts tensor `t` to `target` shape with explicit proof `h`. -/
scoped notation:65 t:max " ▷ " target " ‖ " h:65 => Tensor.broadcast t target h

end Tensor

end Palladium.TensorLib
