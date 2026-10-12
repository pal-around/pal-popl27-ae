{ pkgs }:

let
  julia = pkgs.julia_19-bin;
  aleaRev = "9840424c29a85c47d6aa06a9505597d48a561a43";
  aleaSrc = pkgs.fetchFromGitHub {
    owner = "Tractables";
    repo = "Alea.jl";
    rev = aleaRev;
    hash = "sha256-Bo2RaIpsbRpv3J+AVNFyy42ifmVixqj6g5DKvyOFxO4=";
  };
in
pkgs.writeShellApplication {
  name = "alea-jl";
  runtimeInputs = [ julia pkgs.coreutils ];
  text = ''
    export ALEA_JL_SRC="${aleaSrc}"

    default_cache_root="''${XDG_CACHE_HOME:-$HOME/.cache}/pal-dev/alea-jl"
    cache_root="''${ALEA_JL_CACHE_ROOT:-$default_cache_root}"
    state_file="$cache_root/.setup-${julia.version}-${aleaRev}"
    export JULIA_PROJECT="$cache_root/project"
    export JULIA_DEPOT_PATH="$cache_root/depot"

    mkdir -p "$cache_root" "$JULIA_DEPOT_PATH"

    if [ ! -f "$state_file" ] || [ ! -d "$JULIA_DEPOT_PATH/packages" ]; then
      chmod -R u+w "$JULIA_PROJECT" 2>/dev/null || true
      rm -rf "$JULIA_PROJECT"
      mkdir -p "$JULIA_PROJECT"
      cp "$ALEA_JL_SRC/Project.toml" "$JULIA_PROJECT/Project.toml"
      cp "$ALEA_JL_SRC/Manifest.toml" "$JULIA_PROJECT/Manifest.toml"
      cp -r "$ALEA_JL_SRC/src" "$JULIA_PROJECT/src"
      ln -sfn "$ALEA_JL_SRC/lib" "$JULIA_PROJECT/lib"
      chmod -R u+w "$JULIA_PROJECT"

      (
        cd "$JULIA_PROJECT"
        julia --project="$JULIA_PROJECT" -e '
          using Pkg
          Pkg.instantiate()
          Pkg.develop(path="lib/IRTools.jl")
          Pkg.develop(path="lib/CUDD.jl")
          Pkg.develop(path="lib/Dice.jl")
        '
      )

      touch "$state_file"
    fi

    exec julia --project="$JULIA_PROJECT" "$@"
  '';
}
