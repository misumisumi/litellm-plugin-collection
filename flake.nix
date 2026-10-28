{
  description = "litellm plugin collection (Quinella features)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      pythonVersion = pkgs: pkgs.python3.pythonVersion;

      sitePackages = { pkgs, pkg }: "${pkg}/lib/python${pythonVersion pkgs}/site-packages";

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

      # extraPackages: list of paths or derivations to add to PYTHONPATH
      toPythonPath = { pkgs, items }:
        pkgs.lib.concatStringsSep ":" (map (item:
          if pkgs.lib.isPath item then toString item
          else sitePackages { inherit pkgs; pkg = item; }
        ) items);

      mkLitellmWithPlugin = { pkgs
                            , litellm ? pkgs.litellm
                            , pluginSrc ? ./.
                            , websockets ? pkgs.python3Packages.websockets
                            , extraPackages ? [ ]
                            }:
        let
          pluginPkg = mkPluginPkg { inherit pkgs; };
          pluginSitePackages = sitePackages { inherit pkgs; pkg = pluginPkg; };
          websocketsSitePackages = sitePackages { inherit pkgs; pkg = websockets; };
          extraPath = toPythonPath { inherit pkgs; items = extraPackages; };
          basePythonPath = pkgs.lib.concatStringsSep ":" (pkgs.lib.filter (x: x != "") [
            pluginSitePackages
            websocketsSitePackages
            extraPath
          ]);
          drv = pkgs.writeShellApplication {
            name = "litellm";
            runtimeInputs = [ litellm websockets ] ++ extraPackages;
            text = ''
              export PYTHONPATH="${basePythonPath}''${PYTHONPATH:+:$PYTHONPATH}"
              exec ${litellm}/bin/litellm "$@"
            '';
            passthru = {
              withPlugins = plugins:
                mkLitellmWithPlugin {
                  inherit pkgs litellm pluginSrc websockets;
                  extraPackages = extraPackages ++ plugins;
                };
            };
          };
        in
        drv;
    in {
      packages = forAllSystems (pkgs: {
        default = mkLitellmWithPlugin { inherit pkgs; };
        plugin = mkPluginPkg { inherit pkgs; };
        llmlingua = pkgs.callPackage ./llmlingua.nix { };
      });

      overlays.default = final: prev: {
        litellm-with-plugin-collection = mkLitellmWithPlugin { pkgs = final; };
      };

      lib = {
        inherit mkLitellmWithPlugin mkPluginPkg toPythonPath;
      };
    };
}