import os
import shutil
import subprocess
import tarfile
import urllib.request

import conda.cli.python_api
from conda.cli.python_api import Commands as conda_commands


BASE_ENV = "blas_compare_base"
BASE_PKG_DEPS = (
    "numpy",
    "ninja",
    "pyyaml",
    "setuptools",
    "cmake",
    "cffi",
    "hypothesis",
)

GENERIC_ENV_VARS = ("USE_CUDA=0", "USE_ROCM=0")
MKL_2020_3_PATH = "/tmp/mkl_2020_3/"
SUB_ENVS = {
    "blas_compare_mkl_2020_3": (
        (),
        (),
        ("BLAS=MKL", f"CMAKE_PREFIX_PATH={MKL_2020_3_PATH}") + GENERIC_ENV_VARS,
    ),
    "blas_compare_mkl_2020_2": (
        ("mkl=2020.2", "mkl-include=2020.2"),
        (),
        ("BLAS=MKL",) + GENERIC_ENV_VARS,
    ),
    "blas_compare_mkl_2020_1": (
        ("mkl=2020.1", "mkl-include=2020.1"),
        (),
        ("BLAS=MKL",) + GENERIC_ENV_VARS,
    ),
    "blas_compare_mkl_2020_0": (
        ("mkl=2020.0", "mkl-include=2020.0"),
        (),
        ("BLAS=MKL",) + GENERIC_ENV_VARS,
    ),
    "blas_compare_openblas": (
        ("openblas",),
        (),
        ("BLAS=OpenBLAS",) + GENERIC_ENV_VARS,
    ),
    "blas_compare_blis": (
        (),
        ("conda-forge", ("blis",)),
        ("BLAS=blis", "BLIS_HOME={CONDA_PREFIX}") + GENERIC_ENV_VARS,
    ),
    "blas_compare_eigen": (
        (),
        (),
        ("BLAS=Eigen",) + GENERIC_ENV_VARS,
    ),
}


def get_mkl_2020_3():
    lib_path = "/tmp/l_mkl_2020.3.279.tgz"
    lib_dir_path = lib_path[:-4]
    lib_url = "http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/16903/l_mkl_2020.3.279.tgz"
    if not os.path.exists(lib_path):
        print("Downloading MKL 2020.3")
        urllib.request.urlretrieve(lib_url, lib_path)

    if not os.path.exists(lib_dir_path):
        print("Extracting MKL 2020.3")
        tar = tarfile.open(lib_path, "r:gz")
        tar.extractall(path="/tmp/")
        tar.close()

    if not os.path.exists(MKL_2020_3_PATH):
        print("Installing MKL 2020.3")
        tmp_dir = "/tmp/mkl_2020_3_tmp"
        subprocess.run(
            f"{lib_dir_path}/install.sh --accept_eula --install_dir {tmp_dir} --silent",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        print("Copying include/ and lib/")
        os.makedirs(MKL_2020_3_PATH)
        shutil.copytree(os.path.join(tmp_dir, "mkl/include/"), os.path.join(MKL_2020_3_PATH, "include"))
        shutil.copytree(os.path.join(tmp_dir, "mkl/lib/intel64/"), os.path.join(MKL_2020_3_PATH, "lib"))

        print("Uninstalling MKL 2020.3")
        subprocess.run(
            f"{lib_dir_path}/install.sh --PSET_MODE=uninstall --silent",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def conda_run(*args):
    stdout, stderr, retcode = conda.cli.python_api.run_command(*args)
    if retcode:
        raise OSError(f"conda error: {str(args)}  retcode: {retcode}\n{stderr}")

    return stdout


def prepare():
    get_mkl_2020_3()

    current_envs = conda_run(conda_commands.INFO, "--envs")
    current_envs = [
        i.split()[0] for i in current_envs.strip().splitlines() if i and i[0] != "#"
    ]

    for env in (BASE_ENV,) + tuple(SUB_ENVS.keys()):
        if env in current_envs:
            print(f"Removing old env: {env}")
            conda_run(conda_commands.REMOVE, "--name", env, "--all")

    print(f"Creating env: {BASE_ENV}")
    conda_run(
        conda_commands.CREATE, "--no-default-packages", "--name", BASE_ENV, "python=3"
    )

    base_pkg = conda_run(conda_commands.LIST, "-n", BASE_ENV)
    base_pkg = [
        i.split()[0] for i in base_pkg.strip().splitlines() if i and i[0] != "#"
    ]
    pkg_overlap = set(BASE_PKG_DEPS).intersection(base_pkg)
    if pkg_overlap:
        print("Warning: Base env may not be clean:")
        for pkg in sorted(pkg_overlap):
            print(f"  {pkg} already installed")

    print(f"Testing that `{BASE_ENV}` can be activated.")
    base_source = subprocess.run(
        f"source activate {BASE_ENV}    ",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if base_source.returncode:
        raise OSError(
            "Failed to source base environment:\n"
            f"  stdout: {base_source.stdout.decode('utf-8')}\n"
            f"  stderr: {base_source.stderr.decode('utf-8')}"
        )

    for env, (deps, alt_channel_deps, env_vars) in SUB_ENVS.items():
        print(f"Creating env: {env}")
        conda_run(conda_commands.CREATE, "--name", env, "--clone", BASE_ENV)
        print("Installing packages")
        conda_run(conda_commands.INSTALL, "-n", env, *(BASE_PKG_DEPS + deps))

        if alt_channel_deps:
            channel, channel_deps = alt_channel_deps
            print(f"Installing packages ({channel})")
            conda_run(conda_commands.INSTALL, "-n", env, "-c", channel, *channel_deps)

        if env_vars:
            print("Setting environment variables.")

            prefix = [
                l.split()[1]
                for l in conda_run(conda_commands.INFO, "--envs").replace("*", " ").splitlines()
                if l and l.split()[0] == env
            ][0]
            env_vars = tuple(i.format(CONDA_PREFIX=prefix) for i in env_vars)

            # This does not appear to be possible using the python API.
            env_set = subprocess.run(
                f"source activate {env} && conda env config vars set {' '.join(env_vars)}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if env_set.returncode:
                raise OSError(
                    "Failed to set environment variables:\n"
                    f"  stdout: {env_set.stdout.decode('utf-8')}\n"
                    f"  stderr: {env_set.stderr.decode('utf-8')}"
                )

            actual_env_vars = subprocess.run(
                f"source activate {env} && env",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout.decode("utf-8").strip().splitlines()
            for e in env_vars:
                assert e in actual_env_vars, f"{e} not in envs"

    git_root = subprocess.check_output(
        "git rev-parse --show-toplevel",
        shell=True,
        cwd=os.path.dirname(os.path.realpath(__file__))
    ).decode("utf-8").strip()

    for env in SUB_ENVS.keys():
        print(f"Building PyTorch for env `{env}`.")

        # We have to clean during each build because otherwise BLAS
        # will be reused. As a result build time is non-trivial.
        build_run = subprocess.run(
            f"source activate {env} && "
            f"cd {git_root} && "
            "python setup.py clean && "
            "python setup.py install",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )



def main():
    prepare()


if __name__ == "__main__":
    main()
