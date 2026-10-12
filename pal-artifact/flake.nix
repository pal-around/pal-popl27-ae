{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaCapabilities = [ "7.5" ];
            cudaForwardCompat = false;
            allowUnsupportedSystem = true;
            allowBroken = true;
          };
          overlays = [
            (final: prev: { cudaPackages = prev.cudaPackages_12_2; })
          ];
        };

        jax = if pkgs.stdenv.isLinux then
            (pkgs.python312.pkgs.jax.override { cudaSupport = true; }).overridePythonAttrs (_: { doCheck = false; })
          else
            (pkgs.python312.pkgs.jax.override { jaxlib = pkgs.python312.pkgs.jaxlib; }).overridePythonAttrs (_: { doCheck = false; });
        python312 = pkgs.python312.override {
          packageOverrides = final: prev: {
            jax = jax;
            chex = prev.chex.override { jax = jax; };
            optax = prev.optax.override { jax = jax; chex = final.chex; };
          };
        };

        bngen-upstream = pkgs.callPackage ./nix/bngen-upstream.nix { };
        dice = pkgs.callPackage ./nix/dice-bayestensor/dice.nix { };
        memo = pkgs.callPackage ./nix/memo.nix { python312 = python312; jax = jax; };
        alea-jl = pkgs.callPackage ./nix/alea.jl.nix { };

        # PAL runs from the source tree, so only its Python dependencies are listed here.
        python = python312.withPackages (ps: [
          ps.numpy
          ps.optax
          ps.click
          ps.pytest
          memo
        ]);
        tex = pkgs.texliveSmall.withPackages (ps: with ps; [
          latexmk
          pgfplots
          booktabs
          makecell
          fontawesome5
          xcolor
          siunitx
          adjustbox
          collectbox
          multirow
          xkeyval
        ]);
        # Dice recurses deeply on large programs and overflows the default 8 MB stack;
        # this `dice` runs it with the stack limit raised as far as the system allows.
        diceUnlimitedStack = pkgs.writeShellScriptBin "dice" ''
          ulimit -s unlimited 2>/dev/null || ulimit -s "$(ulimit -H -s)"
          exec ${dice}/bin/dice "$@"
        '';
        tools = [
          python
          diceUnlimitedStack
          bngen-upstream
          alea-jl
          tex
        ] ++ (with pkgs; [
          ruff  # PAL formats the programs it prints (--print-compiled) with it
          gnumake
          coreutils
          gnugrep
          gnused
          gawk
          which
          procps
        ]);
        env = pkgs.buildEnv {
          name = "pal-artifact-env";
          paths = tools;
        };
        shell = pkgs.mkShell {
          packages = tools;
          # Run from the repository root: `import pal` then resolves to its source tree.
          shellHook = ''
            unset NIX_LD NIX_LD_LIBRARY_PATH LD_LIBRARY_PATH
            export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
          '';
        };
      in
      {
        packages = pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          default = env;
          artifact-env = env;
          jax = jax;
          memo = memo;
          python = python;
          dice = dice;
          bngen = bngen-upstream;
          alea-jl = alea-jl;
          tex = tex;
        };
        devShells = {
          default = shell;
          artifact = shell;
        };
      }
    );
}
