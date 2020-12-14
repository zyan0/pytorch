from typing import Dict, Tuple, Union

from tasks.spec import Setup, TaskSpec


# These will be flattened into:
#   FLAT_TASKS: Dict[Tuple[str, ...], TaskSpec]
# However a more lenient structure is allowed here for ease of definition.
KEY_TYPE = Union[str, Tuple[str, ...]]
VALUE_TYPE = Union[TaskSpec, Dict[KEY_TYPE, TaskSpec]]
TASKS: Dict[KEY_TYPE, VALUE_TYPE] = {
    'empty': {
        'no allocation': TaskSpec(
            Setup.NONE,
            r'torch.empty(())',
            r'torch::empty({0});',
        ),

        'with allocation': TaskSpec(
            Setup.NONE,
            r'torch.empty((1,))',
            r'torch::empty({1});',
        ),
    },

    'Pointwise': {
        ('add', 'Tensor-Tensor'): TaskSpec(
            Setup.GENERIC,
            r"x += y_float",
            r"x += y_float;",
            torchscript_args=("x", "y_float"),
        ),

        ('add', 'Tensor-Tensor (type promotion)'): TaskSpec(
            Setup.GENERIC,
            r"x += y_int",
            r"x += y_int;",
        ),

        ('add', 'Tensor-Tensor (out of place)'): TaskSpec(
            Setup.GENERIC,
            r"x + y_float",
            r"x + y_float;",
        ),

        'zero_': TaskSpec(
            Setup.GENERIC,
            r"x.zero_()",
            r"x.zero_();",
        ),

        ('equality', 'Tensor-Tensor'): TaskSpec(
            Setup.GENERIC,
            r"x == y_float",
            r"x == y_float;"
        ),

        ('equality', 'Tensor-Scalar'): TaskSpec(
            Setup.GENERIC,
            r"x == 1.0",
            r"x == 1.0;"
        ),
    },

    'MatMul': TaskSpec(
        Setup.GENERIC,
        r"z = torch.mm(x, y_float)",
        r"auto z = torch::mm(x, y_float);",
        torchscript_args=("x", "y_float"),
        torchscript_return='z',
    ),

    'Mesoscale': {
        'MatMul-Bias-ReLU': TaskSpec(
            Setup.MESOSCALE,
            r"z = torch.nn.functional.relu(torch.mm(x, y) + bias)",
            r"auto z = torch::nn::functional::relu(torch::mm(x, y) + bias);",
            torchscript_args=("x", "y", "bias"),
            torchscript_return='z',
        ),

        'Off diagonal indices': TaskSpec(
            Setup.MESOSCALE,
            r"z = torch.arange(eye_4.numel())[eye_4.flatten() == 0]",
            r"auto z = torch::arange(eye_4.numel()).index({eye_4.flatten() == 0});",
            torchscript_args=("eye_4",),
            torchscript_return='z',
        ),
    },

    'AutoGrad': {
        'simple': TaskSpec(
            Setup.AUTOGRAD,
            r"""
            y = torch.nn.functional.relu(x * w0) * w1
            y.backward()
            """,
            r"""
            auto y = torch::nn::functional::relu(x * w0) * w1;
            y.backward();
            """,
            torchscript_args=("x", "w0", "w1"),
        ),

        'intermediate': TaskSpec(
            Setup.AUTOGRAD,
            r"""
            branch_0 = torch.nn.functional.gelu(x * w0)
            branch_1 = torch.nn.functional.prelu(y, w1)
            z = torch.nn.functional.normalize(
                torch.cat([branch_0, branch_1]),
                p=2.0, dim=0,
            ).dot(w2)
            z.backward()
            """,
            r"""
            auto branch_0 = torch::nn::functional::gelu(x * w0);
            auto branch_1 = torch::nn::functional::prelu(x, w0);
            auto z = torch::nn::functional::normalize(
                torch::cat({branch_0, branch_1}),
                torch::nn::functional::NormalizeFuncOptions().p(2).dim(0)
            ).dot(w2);
            z.backward();
            """,
            torchscript_args=("x", "y", "w0", "w1", "w2"),
        ),
    },
}


def flatten(subtree: Dict[KEY_TYPE, VALUE_TYPE], prefix=(), result=None):
    if result is None:
        result = {}

    for k, v in subtree.items():
        k = k if isinstance(k, tuple) else (k,)
        if isinstance(v, TaskSpec):
            result[prefix + k] = v
        else:
            flatten(v, prefix + k, result)

    return result

FLAT_TASKS: Dict[Tuple[str, ...], TaskSpec] = flatten(TASKS)
