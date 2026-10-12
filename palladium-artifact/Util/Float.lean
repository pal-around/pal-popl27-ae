/-! # Total division on floats

Normalization in the source language is total (paper, §"PAL: Semantics"): when a
division of zero by zero occurs, the resulting mass is zero.
`Float.div0` implements this convention and is the division used by both PAL and
Jay. For every other pair of operands it is IEEE-754 division.
-/

/-- Division with `0 / 0 = 0`; IEEE-754 division otherwise. -/
def Float.div0 (a b : Float) : Float :=
  if a == 0 && b == 0 then 0 else a / b
