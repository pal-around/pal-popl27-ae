import Palladium.Compiler.Encoding.ContextShape
import Palladium.Compiler.Encoding.EnvironmentIndices
import Palladium.Compiler.Encoding.ToIndicesBridge

/-!
# Encoding PAL contexts as tensor shapes and indices

This module defines the encoding between PAL random contexts and tensor
representations used by the compiler:
- `RandomContext.toShape`: each discrete variable contributes a dimension
- `Environment.toIndices`: extract tensor indices from a random environment
- Bridge lemmas connecting indices across context operations (insert, unpermute)
-/
