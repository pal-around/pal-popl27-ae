import TensorLib.Shape.Basics

namespace Palladium.TensorLib
namespace Shape

/-! ## Squeezable -/

/-- `Squeezable σₛ axis` means the dimension at `axis` is 1.
    The output shape is always `σₛ.slice axis`. -/
def Squeezable (σₛ : Shape) (axis : Fin σₛ.rank) : Prop :=
  σₛ[axis.val]'axis.isLt = 1

instance (σₛ : Shape) (axis : Fin σₛ.rank) : Decidable (Squeezable σₛ axis) :=
  inferInstanceAs (Decidable (σₛ[axis.val]'axis.isLt = 1))

end Shape
end Palladium.TensorLib
