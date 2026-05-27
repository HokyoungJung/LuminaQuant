"""Build native metric backends (C and Rust) across platforms."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(
    command: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> tuple[int, str, str]:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, env=env)
    return int(result.returncode), str(result.stdout or ""), str(result.stderr or "")


def _native_lib_filename(stem: str) -> str:
    system_name = platform.system().lower()
    if system_name == "windows":
        return f"{stem}.dll"
    if system_name == "darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"


def _detect_vs_install() -> str:
    candidates = [
        Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"),
        Path("C:/Program Files/Microsoft Visual Studio/Installer/vswhere.exe"),
    ]
    for vswhere in candidates:
        if not vswhere.exists():
            continue
        rc, out, _ = _run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            cwd=Path.cwd(),
        )
        if rc == 0 and out.strip():
            return out.strip().splitlines()[0].strip()
    return ""


def _build_c_windows(root: Path) -> tuple[bool, Path | None]:
    src = root / "native" / "c_metrics" / "evaluate_metrics.c"
    out_dir = root / "native" / "c_metrics" / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    dll = out_dir / _native_lib_filename("lumina_metrics")
    if not src.exists():
        return False, None

    env = os.environ.copy()
    vs_install = env.get("VSINSTALLDIR", "").strip() or _detect_vs_install()
    if not vs_install:
        print("[C] build rc: 1")
        print("[ERROR] Visual Studio Build Tools not found.")
        return False, None

    devcmd = Path(vs_install) / "Common7" / "Tools" / "VsDevCmd.bat"
    if not devcmd.exists():
        print("[C] build rc: 1")
        print(f"[ERROR] VsDevCmd not found: {devcmd}")
        return False, None

    runner = out_dir / "build_direct.bat"
    runner.write_text(
        "\r\n".join(
            [
                "@echo off",
                f'call "{devcmd}" -no_logo -arch=x64 -host_arch=x64',
                f'cl /nologo /O2 /LD "{src}" /link /OUT:"{dll}"',
            ]
        )
        + "\r\n",
        encoding="utf-8",
    )

    rc, out, err = _run(["cmd.exe", "/d", "/c", str(runner)], cwd=out_dir, env=env)
    try:
        runner.unlink(missing_ok=True)
    except Exception:
        pass

    print("[C] build rc:", rc)
    if out.strip():
        print(out.strip())
    if err.strip():
        print(err.strip())
    return rc == 0 and dll.exists(), (dll if dll.exists() else None)


def _build_c_posix(root: Path) -> tuple[bool, Path | None]:
    src = root / "native" / "c_metrics" / "evaluate_metrics.c"
    out_dir = root / "native" / "c_metrics" / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    lib_path = out_dir / _native_lib_filename("lumina_metrics")
    if not src.exists():
        return False, None

    compiler = str(os.getenv("CC", "")).strip()
    if not compiler:
        compiler = shutil.which("clang") or shutil.which("gcc") or ""
    if not compiler:
        print("[C] build rc: 1")
        print("[ERROR] No C compiler found (clang/gcc)")
        return False, None

    system_name = platform.system().lower()
    if system_name == "darwin":
        cmd = [compiler, "-O3", "-fPIC", "-dynamiclib", str(src), "-o", str(lib_path)]
    else:
        cmd = [compiler, "-O3", "-fPIC", "-shared", str(src), "-o", str(lib_path), "-lm"]

    rc, out, err = _run(cmd, cwd=out_dir, env=os.environ.copy())
    print("[C] build rc:", rc)
    if out.strip():
        print(out.strip())
    if err.strip():
        print(err.strip())
    return rc == 0 and lib_path.exists(), (lib_path if lib_path.exists() else None)


def _build_c(root: Path) -> tuple[bool, Path | None]:
    if platform.system().lower() == "windows":
        return _build_c_windows(root)
    return _build_c_posix(root)


def _build_rust(root: Path) -> tuple[bool, Path | None]:
    rust_dir = root / "native" / "rust_metrics"
    return _build_rust_dir(rust_dir, "lumina_metrics")


def _build_rust_dir(rust_dir: Path, stem: str) -> tuple[bool, Path | None]:
    if not rust_dir.exists():
        return False, None

    env = os.environ.copy()
    cargo_bin = str(Path.home() / ".cargo" / "bin")
    env["PATH"] = cargo_bin + os.pathsep + env.get("PATH", "")
    cargo_exec = shutil.which("cargo", path=env.get("PATH"))
    if not cargo_exec:
        print("[RUST] build rc: 1")
        print("[ERROR] cargo not found. Install Rust toolchain first.")
        return False, None

    rc, out, err = _run([cargo_exec, "build", "--release"], cwd=rust_dir, env=env)
    print("[RUST] build rc:", rc)
    if out.strip():
        print(out.strip())
    if err.strip():
        print(err.strip())

    lib_path = rust_dir / "target" / "release" / _native_lib_filename(stem)
    return rc == 0 and lib_path.exists(), (lib_path if lib_path.exists() else None)


def _build_rust_rawfirst(root: Path) -> tuple[bool, Path | None]:
    rust_dir = root / "native" / "rust_rawfirst"
    return _build_rust_dir(rust_dir, "lumina_rawfirst")


def _build_rust_hybrid_optuna(root: Path) -> tuple[bool, Path | None]:
    rust_dir = root / "native" / "rust_hybrid_optuna"
    return _build_rust_dir(rust_dir, "lumina_hybrid_optuna")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LuminaQuant native backends")
    parser.add_argument(
        "--backend",
        choices=["all", "c", "rust", "rust-rawfirst", "rust-hybrid-optuna"],
        default="all",
        help="Which backend(s) to build",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_c = args.backend in {"all", "c"}
    build_rust = args.backend in {"all", "rust"}
    build_rust_rawfirst = args.backend in {"all", "rust-rawfirst"}
    build_rust_hybrid_optuna = args.backend in {"all", "rust-hybrid-optuna"}

    c_ok = False
    rust_ok = False
    rust_rawfirst_ok = False
    rust_hybrid_optuna_ok = False
    c_lib: Path | None = None
    rust_lib: Path | None = None
    rust_rawfirst_lib: Path | None = None
    rust_hybrid_optuna_lib: Path | None = None

    if build_c:
        c_ok, c_lib = _build_c(root)
    if build_rust:
        rust_ok, rust_lib = _build_rust(root)
    if build_rust_rawfirst:
        rust_rawfirst_ok, rust_rawfirst_lib = _build_rust_rawfirst(root)
    if build_rust_hybrid_optuna:
        rust_hybrid_optuna_ok, rust_hybrid_optuna_lib = _build_rust_hybrid_optuna(root)

    print("build_native_backends summary")
    if build_c:
        print(f"c_ok={c_ok} c_lib={c_lib}")
    if build_rust:
        print(f"rust_ok={rust_ok} rust_lib={rust_lib}")
    if build_rust_rawfirst:
        print(f"rust_rawfirst_ok={rust_rawfirst_ok} rust_rawfirst_lib={rust_rawfirst_lib}")
    if build_rust_hybrid_optuna:
        print(
            "rust_hybrid_optuna_ok="
            f"{rust_hybrid_optuna_ok} rust_hybrid_optuna_lib={rust_hybrid_optuna_lib}"
        )

    if (
        (build_c and not c_ok)
        or (build_rust and not rust_ok)
        or (build_rust_rawfirst and not rust_rawfirst_ok)
        or (build_rust_hybrid_optuna and not rust_hybrid_optuna_ok)
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
