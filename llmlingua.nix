{ lib
, python3Packages
, fetchPypi
}:

python3Packages.buildPythonPackage rec {
  pname = "llmlingua";
  version = "0.2.2";

  src = fetchPypi {
    inherit pname version;
    sha256 = "0wl50pwp7580qsqygfawbssc9czmlvynpnyshlm52rasipfsw30s";
  };

  format = "setuptools";

  propagatedBuildInputs = with python3Packages; [
    transformers
    accelerate
    torch
    tiktoken
    nltk
    numpy
  ];

  pythonImportsCheck = [ "llmlingua" ];

  meta = with lib; {
    description = "Prompt compression for LLMs";
    homepage = "https://github.com/microsoft/LLMLingua";
    license = licenses.mit;
  };
}