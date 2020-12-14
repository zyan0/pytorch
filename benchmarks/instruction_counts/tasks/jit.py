import importlib
import itertools as it
import os
import textwrap
from typing import Optional, Tuple

from torch.utils.benchmark import Language

from tasks.spec import TaskSpec


TORCHSCRIPT_DEFINITION_TEMPLATE = """\
import torch

@torch.jit.script
def f({args_str}):
{stmt_str}
    return {return_str}
"""


TORCHSCRIPT_PYTHON_SETUP_TEMPLATE = """\
{setup_str}
f = torch.jit.load({model_path_str})

# Warmup `f`
for _ in range(3):
    f({args_str})
"""


TORCHSCRIPT_CPP_SETUP_TEMPLATE = """\
#include <string>
#include <vector>
#include <torch/script.h>

{setup_str}

const std::string fpath = "{model_path}";
auto module = torch::jit::load(fpath);

// Warmup `module`
for (int i = 0; i < 3; i++) {{
{stmt}
}}
"""


def wrap_in_jit_script(spec: TaskSpec) -> Optional[str]:
    if not spec.torchscript_args:
        return

    return TORCHSCRIPT_DEFINITION_TEMPLATE.format(
        args_str=', '.join(spec.torchscript_args),
        stmt_str=textwrap.indent(spec.py_stmt, ' ' * 4),
        return_str=spec.torchscript_return or ''
    )


def generate_torchscript_file(
    spec: TaskSpec,
    name: str,
    working_dir: str,
) -> Optional[str]:
    src = wrap_in_jit_script(spec)
    if src is None:
        return

    # TorchScript requires an actual source file, so we have to write a file
    # rather than simply evaling a definition string.
    module_path = os.path.join(working_dir, f"ts_{name}.py")
    if os.path.exists(module_path):
        # This will only happen if two entries in FLAT_TASKS have very similar
        # names, and collide during regularization.
        raise ValueError(f"File {module_path} already exists.")

    with open(module_path, "wt") as f:
        f.write(src)

    module_spec = importlib.util.spec_from_file_location(f"ts_{name}", module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    f = module.f

    model_path = os.path.join(working_dir, f"ts_{name}.pt")
    f.save(model_path)
    os.remove(module_path)

    return model_path


def _make_python_stmt_and_setup(
    setup: str,
    model_path: str,
    torchscript_args: Tuple[str, ...]
) -> Tuple[str, str]:
    args_str = ', '.join(torchscript_args)
    return f'f({args_str})', TORCHSCRIPT_PYTHON_SETUP_TEMPLATE.format(
        setup_str=setup,
        model_path_str=repr(model_path),
        args_str=args_str,
    )


def _make_cpp_stmt_and_setup(
    setup: str,
    model_path: str,
    torchscript_args: Tuple[str, ...]
) -> Tuple[str, str]:
    stmt = "\n".join(it.chain(
        ['std::vector<torch::jit::IValue> inputs;'],
        [f'inputs.push_back({arg});' for arg in torchscript_args],
        ['module.forward(inputs);']
    ))
    assert '"' not in model_path  # This would not be escaped correctly.
    return stmt, TORCHSCRIPT_CPP_SETUP_TEMPLATE.format(
        setup_str=setup,
        model_path=model_path,
        stmt=textwrap.indent(stmt, ' ' * 4)
    )


def make_stmt_and_setup(
    language: Language,
    setup: str,
    model_path: str,
    torchscript_args: Tuple[str, ...]
) -> Tuple[str, str]:
    if language == Language.PYTHON:
        return _make_python_stmt_and_setup(setup, model_path, torchscript_args)
    else:
        assert language == Language.CPP
        return _make_cpp_stmt_and_setup(setup, model_path, torchscript_args)
