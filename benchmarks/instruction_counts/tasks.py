# import atexit
# import dataclasses
# import enum
# import importlib.util
# import os
# import shutil
# import textwrap
# import tempfile
# from typing import Dict, Iterable, Optional, Tuple, Union


# from torch.utils.benchmark import Timer, Language


# class Setup(enum.Enum):
#     NONE = 0
#     GENERIC = 1
#     MESOSCALE = 2
#     AUTOGRAD = 3


# @dataclasses.dataclass()
# class TaskSpec:
#     setup: Setup
#     py_stmt: str
#     cpp_stmt: str
#     torchscript_args: Optional[Tuple[str, ...]] = None
#     torchscript_return: Optional[str] = None

#     def __post_init__(self):
#         self.py_stmt = textwrap.dedent(self.py_stmt).strip()
#         self.cpp_stmt = textwrap.dedent(self.cpp_stmt).strip()


# # These will be flattened into:
# #   Dict[Tuple[str, ...], Task]
# # However a more lenient structure is allowed here for ease of definition.
# _TASKS: Dict[str, Union[TaskSpec, Dict[Union[str, Tuple[str, ...]], TaskSpec]]] = {
#     'empty': {
#         'no allocation': TaskSpec(
#             Setup.NONE,
#             r'torch.empty(())',
#             r'torch::empty({0});',
#         ),

#         'with allocation': TaskSpec(
#             Setup.NONE,
#             r'torch.empty((1,))',
#             r'torch::empty({1});',
#         ),
#     },

#     'Pointwise': {
#         ('add', 'Tensor-Tensor'): TaskSpec(
#             Setup.GENERIC,
#             r"x += y_float",
#             r"x += y_float;",
#             torchscript_args=("x", "y_float"),
#         ),

#         ('add', 'Tensor-Tensor (type promotion)'): TaskSpec(
#             Setup.GENERIC,
#             r"x += y_int",
#             r"x += y_int;",
#         ),

#         ('add', 'Tensor-Tensor (out of place)'): TaskSpec(
#             Setup.GENERIC,
#             r"x + y_float",
#             r"x + y_float;",
#         ),

#         'zero_': TaskSpec(
#             Setup.GENERIC,
#             r"x.zero_()",
#             r"x.zero_();",
#         ),

#         ('equality', 'Tensor-Tensor'): TaskSpec(
#             Setup.GENERIC,
#             r"x == y_float",
#             r"x == y_float;"
#         ),

#         ('equality', 'Tensor-Scalar'): TaskSpec(
#             Setup.GENERIC,
#             r"x == 1.0",
#             r"x == 1.0;"
#         ),
#     },

#     'MatMul': TaskSpec(
#         Setup.GENERIC,
#         r"z = torch.mm(x, y_float)",
#         r"auto z = torch::mm(x, y_float);",
#         torchscript_args=("x", "y_float"),
#         torchscript_return='z',
#     ),

#     'Mesoscale': {
#         'MatMul-Bias-ReLU': TaskSpec(
#             Setup.MESOSCALE,
#             r"z = torch.nn.functional.relu(torch.mm(x, y) + bias)",
#             r"auto z = torch::nn::functional::relu(torch::mm(x, y) + bias);",
#             torchscript_args=("x", "y", "bias"),
#             torchscript_return='z',
#         ),

#         'Off diagonal indices': TaskSpec(
#             Setup.MESOSCALE,
#             r"z = torch.arange(eye_4.numel())[eye_4.flatten() == 0]",
#             r"auto z = torch::arange(eye_4.numel()).index({eye_4.flatten() == 0});",
#             torchscript_args=("eye_4",),
#             torchscript_return='z',
#         ),
#     },

#     'AutoGrad': {
#         'simple': TaskSpec(
#             Setup.AUTOGRAD,
#             r"""
#             y = torch.nn.functional.relu(x * w0) * w1
#             y.backward()
#             """,
#             r"""
#             auto y = torch::nn::functional::relu(x * w0) * w1;
#             y.backward();
#             """,
#             torchscript_args=("x", "w0", "w1"),
#         ),

#         'intermediate': TaskSpec(
#             Setup.AUTOGRAD,
#             r"""
#             branch_0 = torch.nn.functional.gelu(x * w0)
#             branch_1 = torch.nn.functional.prelu(y, w1)
#             z = torch.nn.functional.normalize(
#                 torch.cat([branch_0, branch_1]),
#                 p=2.0, dim=0,
#             ).dot(w2)
#             z.backward()
#             """,
#             r"""
#             auto branch_0 = torch::nn::functional::gelu(x * w0);
#             auto branch_1 = torch::nn::functional::prelu(x, w0);
#             auto z = torch::nn::functional::normalize(
#                 torch::cat({branch_0, branch_1}),
#                 torch::nn::functional::NormalizeFuncOptions().p(2).dim(0)
#             ).dot(w2);
#             z.backward();
#             """,
#             torchscript_args=("x", "y", "w0", "w1", "w2"),
#         ),
#     },
# }


# _SetupMap = {
#     Setup.NONE: ("pass", ""),

#     Setup.GENERIC: (
#         textwrap.dedent(r"""
#             x = torch.ones((4, 4))
#             y_float = torch.ones((4, 4))
#             y_int = torch.ones((4, 4), dtype=torch.int32)
#         """).strip(),
#         textwrap.dedent(r"""
#             auto x = torch::ones({4, 4});
#             auto y_float = torch::ones({4, 4});
#             auto y_int = torch::ones({4, 4}, at::kInt);
#         """).strip(),
#     ),

#     Setup.MESOSCALE: (
#         textwrap.dedent(r"""
#             x = torch.ones((4, 4))
#             y = torch.ones((4, 4))
#             bias = torch.ones((1,))

#             eye_4 = torch.eye(4)
#         """).strip(),
#         textwrap.dedent(r"""
#             auto x = torch::ones({4, 4});
#             auto y = torch::ones({4, 4});
#             auto bias = torch::ones({1});

#             auto eye_4 = torch::eye(4);
#         """).strip(),
#     ),

#     Setup.AUTOGRAD: (
#         textwrap.dedent(r"""
#             x = torch.ones((1,))
#             y = torch.ones((1,))
#             w0 = torch.ones((1,), requires_grad=True)
#             w1 = torch.ones((1,), requires_grad=True)
#             w2 = torch.ones((2,), requires_grad=True)
#         """).strip(),
#         textwrap.dedent(r"""
#             auto x = torch::ones({1});
#             auto y = torch::ones({1});

#             auto w0 = torch::ones({1});
#             w0.set_requires_grad(true);

#             auto w1 = torch::ones({1});
#             w1.set_requires_grad(true);

#             auto w2 = torch::ones({2});
#             w2.set_requires_grad(true);
#         """).strip(),
#     ),
# }


# def _torchscript_src(task_spec: TaskSpec):
#     return f"""\
# import torch

# @torch.jit.script
# def f({", ".join(task_spec.torchscript_args)}):
# {textwrap.indent(task_spec.py_stmt, ' ' * 4)}
#     return {task_spec.torchscript_return or ''}
# """


# @dataclasses.dataclass(frozen=True)
# class TimerSpec:
#     stmt: str
#     setup: str
#     language: Language

# def make_timer(spec: TimerSpec):
#     return Timer(stmt=spec.stmt, setup=spec.setup, language=spec.language)


# class Mode(enum.Enum):
#     PY = "Python"
#     CPP = "C++"
#     PY_TS = "Python (TorchScript)"
#     CPP_TS = "C++ (TorchScript)"


# @dataclasses.dataclass()
# class Task:
#     sub_tasks: Dict[Mode, TimerSpec]

#     def __init__(self, key: Tuple[str, ...], task_spec: TaskSpec, working_dir: str, i: int):
#         self.sub_tasks = {}
#         self.sub_tasks[Mode.PY] = TimerSpec(
#             stmt=task_spec.py_stmt,
#             setup=_SetupMap[task_spec.setup][0],
#             language=Language.PYTHON,
#         )

#         self.sub_tasks[Mode.CPP] = TimerSpec(
#             stmt=task_spec.cpp_stmt,
#             setup=_SetupMap[task_spec.setup][1],
#             language=Language.CPP,
#         )

#         if task_spec.torchscript_args is not None:
#             module_path = os.path.join(working_dir, f"ts_{i}.py")
#             with open(module_path, "wt") as f:
#                 f.write(_torchscript_src(task_spec))

#             spec = importlib.util.spec_from_file_location(f"ts_{i}", module_path)
#             ts_module = importlib.util.module_from_spec(spec)
#             spec.loader.exec_module(ts_module)

#             ts_path = os.path.join(working_dir, f"ts_{i}.pt")
#             ts_module.f.save(ts_path)

#             stmt = f'f({", ".join(task_spec.torchscript_args)})  # {key}'
#             self.sub_tasks[Mode.PY_TS] = TimerSpec(
#                 stmt=stmt,
#                 setup=f"""
# {_SetupMap[task_spec.setup][0]}
# f = torch.jit.load({repr(ts_path)})

# # Warmup `f`
# for _ in range(3):
# {textwrap.indent(stmt, ' ' * 4)}
# """,
#                 language=Language.PYTHON,
#             )

#             cpp_stmt_lines = ['std::vector<torch::jit::IValue> inputs;']
#             for i in task_spec.torchscript_args:
#                 cpp_stmt_lines.append(f'inputs.push_back({i});')
#             cpp_stmt_lines.append('module.forward(inputs);')
#             stmt = '\n'.join(cpp_stmt_lines)

#             self.sub_tasks[Mode.CPP_TS] = TimerSpec(
#                 stmt=stmt,
#                 setup=f"""\
# #include <string>
# #include <vector>
# #include <torch/script.h>

# {_SetupMap[task_spec.setup][1]}

# const std::string fpath = "{ts_path}";
# auto module = torch::jit::load(fpath);

# // Warmup `module`
# for (int i = 0; i < 3; i++) {{
# {textwrap.indent(stmt, ' ' * 4)}
# }}

# """,
#                 language=Language.CPP,
#             )


# class Tasks:
#     def __init__(self):
#         self._working_dir = tempfile.mkdtemp()
#         atexit.register(shutil.rmtree, self._working_dir)

#         self._tasks: Dict[Tuple[str, ...], Task] = {}
#         self._add_tasks((), _TASKS)

#     def _add_tasks(self, k, v):
#         if isinstance(v, TaskSpec):
#             self._tasks[k] = Task(k, v, self._working_dir, len(self._tasks))
#         else:
#             assert isinstance(v, dict)
#             for ki, vi in v.items():
#                 if isinstance(ki, str):
#                     ki = (ki,)
#                 assert isinstance(ki, tuple) and ki
#                 assert all(isinstance(kij, str) for kij in ki)
#                 self._add_tasks(k + ki, vi)

#     def items(self):
#         for k, v in self._tasks.items():
#             yield k, v
