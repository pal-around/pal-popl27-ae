{ lib, fetchFromGitHub, pkgs, stdenv }:

let
  rustPlatform = pkgs.makeRustPlatform {
    cargo = pkgs.cargo;
    rustc = pkgs.rustc;
  };
in
rustPlatform.buildRustPackage rec {
  pname = "rsdd";
  version = "fcb5726";

  src = fetchFromGitHub {
    owner = "SHoltzen";
    repo = "rsdd";
    rev = "fcb5726952b39bdc224623603c53d4e7d4326310";
    hash = "sha256-fjDRO3ldS9UXFyHBw/dxZBL47PSateQArox/ccvdhsU=";
  };

  prePatch = ''
    cp ${./Cargo.lock} Cargo.lock
  '';

  cargoPatches = [ ./rsdd.patch ];
  cargoLock = {
      lockFile = ./Cargo.lock;
  };
  cargoBuildFlags = [ "--lib" ];
  doCheck = false;

  installPhase = ''
    runHook preInstall
    artifacts_dir=target/${stdenv.hostPlatform.rust.rustcTargetSpec}/release
    if [ ! -e "$artifacts_dir/librsdd.a" ]; then
      artifacts_dir=target/release
    fi
    mkdir -p $out/lib
    cp "$artifacts_dir/librsdd.a" $out/lib/
    cp "$artifacts_dir/librsdd${stdenv.hostPlatform.extensions.sharedLibrary}" $out/lib/
    runHook postInstall
  '';

  meta = {
    platforms = lib.platforms.linux;
  };
}
