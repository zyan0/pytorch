"""Default set of benchmarks."""

from core.api import Setup, TimerArgs, GroupedTimerArgs
from core.types import FlatIntermediateDefinition
from core.utils import flatten, from_string

BENCHMARKS: FlatIntermediateDefinition = flatten({
    "empty": {
        "no allocation": GroupedTimerArgs(
            Setup.NONE,
            r"torch.empty(())",
            r"torch::empty({0});",
        ),

        "with allocation": GroupedTimerArgs(
            Setup.NONE,
            r"torch.empty((1,))",
            r"torch::empty({1});",
        ),
    },

    "Pointwise": {
        "add": {
            "Tensor-Tensor": GroupedTimerArgs(
                Setup.GENERIC,
                r"x += y_float",
                r"x += y_float;",
                signature=r"f(x, y_float) -> None",
            ),

            "Tensor-Tensor (type promotion)": GroupedTimerArgs(
                Setup.GENERIC,
                r"x += y_int",
                r"x += y_int;",
            ),

            "Tensor-Tensor (out of place)": GroupedTimerArgs(
                Setup.GENERIC,
                r"x + y_float",
                r"x + y_float;",
            ),
        },

        "equality": {
            "Tensor-Tensor": GroupedTimerArgs(
                Setup.GENERIC,
                r"x == y_float",
                r"x == y_float;",
            ),

            "Tensor-Scalar": GroupedTimerArgs(
                Setup.GENERIC,
                r"x == 1.0",
                r"x == 1.0;",
            ),
        },

        "zero_": GroupedTimerArgs(
            Setup.GENERIC,
            r"x.zero_()",
            r"x.zero_();",
        ),
    },

    "MatMul": GroupedTimerArgs(
        Setup.GENERIC,
        r"z = torch.mm(x, y_float)",
        r"auto z = torch::mm(x, y_float);",
    ),

    "Mesoscale": {
        "MatMul-Bias-ReLU": from_string(
            Setup.MESOSCALE,
            r"""
                Python                                   | C++
                ---------------------------------------- | ----------------------------------------
                z0 = torch.mm(x, y) + bias               | auto z0 = torch::mm(x, y) + bias;
                z1 = torch.nn.functional.relu(z0)        | auto z1 = torch::nn::functional::relu(z0);
            """,
            signature=r"f(x, y, bias) -> z1",
        ),

        "Off diagonal indices": from_string(
            Setup.MESOSCALE,
            r"""
                Python                                   | C++
                ---------------------------------------- | ----------------------------------------
                indices = torch.arange(eye_4.numel())    | auto indices = torch::arange(eye_4.numel());
                z = indices[eye_4.flatten() == 0]        | auto z = indices.index({eye_4.flatten() == 0});
            """,
            signature=r"f(eye_4) -> z",
        ),
    },

    "AutoGrad": {
        "simple": from_string(
            Setup.AUTOGRAD,
            r"""
                Python                                   | C++
                ---------------------------------------- | ----------------------------------------
                a0 = torch.nn.functional.relu(x * w0)    | auto a0 = torch::nn::functional::relu(x * w0);
                y = a0 * w1                              | auto y = a0 * w1;
                y.backward()                             | y.backward();
            """,
            num_threads=(1, 2),
            signature=r"f(x, w0, w1) -> None",
        ),

        "intermediate": from_string(
            Setup.AUTOGRAD,
            r"""
                Python                                   | C++
                ---------------------------------------- | ----------------------------------------
                a0 = torch.nn.functional.gelu(x * w0)    | auto a0 = torch::nn::functional::gelu(x * w0);
                a1 = torch.nn.functional.prelu(y, w1)    | auto a1 = torch::nn::functional::prelu(y, w1);
                z = torch.nn.functional.normalize(       | auto z = torch::nn::functional::normalize(
                    torch.cat([a0, a1]),                 |     torch::cat({a0, a1}),
                    p=2.0, dim=0,                        |     torch::nn::functional::NormalizeFuncOptions().p(2).dim(0)
                ).dot(w2)                                | ).dot(w2);
                z.backward()                             | z.backward();
            """,
            num_threads=(1, 2),
            signature=r"f(x, y, w0, w1, w2) -> None",
        )
    }
})
