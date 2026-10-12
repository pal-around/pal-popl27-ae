{ lib
, fetchFromGitHub
, python312
, jax ? python312.pkgs.jax
}:
python312.pkgs.buildPythonPackage rec {
  pname = "memo-lang";
  version = "1.2.10";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "kach";
    repo = "memo";
    rev = "878e3549bd81ef99baebea483d0f696def495db0";
    sha256 = "01ynzybcf839xxjlmb4y6zw8q48nvdld02frwpg5h3579jcc1rh3";
  };

  nativeBuildInputs = with python312.pkgs; [
    setuptools
  ];

  patches = [
    ./memo-python312-compat.patch
  ];

  propagatedBuildInputs = [
    jax
  ];

  pythonImportsCheck = [ "memo" ];

  doCheck = false;

  meta = with lib; {
    description = "Probabilistic programming language for reasoning about reasoning";
    homepage = "https://github.com/kach/memo";
    license = licenses.mit;
  };
}
