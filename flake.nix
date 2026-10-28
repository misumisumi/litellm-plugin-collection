{
  description = "litellm plugin collection (Quinella features)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      pythonVersion = pkgs: pkgs.python3.pythonVersion;

      # Build the plugin collection as a proper Python package (separate build).
      mkPluginPkg = { pkgs }:
        pkgs.python3Packages.buildPythonPackage {
          pname = "litellm-plugin-collection";
          version = "0.1.0";
          src = ./.;
          format = "pyproject";
          propagatedBuildInputs = [ pkgs.python3Packages.litellm ];
          nativeBuildInputs = [ pkgs.python3Packages.setuptools ];
          doCheck = false;
        };

      # Wrap pkgs.litellm so the plugin package is importable at runtime.
      mkLitellmWithPlugin = { pkgs, litellm ? pkgs.litellm }:
        let
          pluginPkg = mkPluginPkg { inherit pkgs; };
          pluginSitePackages = "${pluginPkg}/lib/python${pythonVersion pkgs}/site-packages";
        in
        pkgs.writeShellApplication {
          name = "litellm";
          runtimeInputs = [ litellm pkgs.python3Packages.websockets ];
          text = ''
            export PYTHONPATH="${pluginSitePackages}''${PYTHONPATH:+:$PYTHONPATH}"
            exec ${litellm}/bin/litellm "$@"
          '';
        };
    in {
      packages = forAllSystems (pkgs: {
        default = mkLitellmWithPlugin { inherit pkgs; } // {
          meta = with pkgs.lib; {
            description = "litellm with Quinella plugin collection";
            license = licenses.mit;
            platforms = platforms.all;
          };
        };
        plugin = mkPluginPkg { inherit pkgs; };
        llmlingua = pkgs.callPackage ./llmlingua.nix { };
      });

      overlays.default = final: prev: {
        litellm-with-plugin-collection = mkLitellmWithPlugin { pkgs = final; };
      };

      lib = {
        inherit mkLitellmWithPlugin mkPluginPkg;
      };
    };
}