from typing import Dict, TYPE_CHECKING

from core.api import Setup, GroupedTimerArgs
from core.utils import from_string

if TYPE_CHECKING:
    # See core.api for an explanation.
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language


def _setup_from_string(schema: str) -> Dict[Language, str]:
    # Use the same underlying parser as `from_string`
    g: GroupedTimerArgs = from_string(setup=Setup.NONE, stmts=schema)
    assert g.torchscript_signature is None
    return {
        Language.PYTHON: g.py_stmt or "",
        Language.CPP: g.cpp_stmt or "",
    }


SETUP_MAP: Dict[Setup, Dict[Language, str]] = {
    Setup.NONE: {
        Language.PYTHON: "pass",
        Language.CPP: "",
    },

    Setup.GENERIC: _setup_from_string(
        r"""
            Python                                   | C++
            ---------------------------------------- | ----------------------------------------
            x = torch.ones((4, 4))                   | auto x = torch::ones({4, 4});
            y_float = torch.ones((4, 4))             | auto y_float = torch::ones({4, 4});
            y_int = torch.ones(                      | auto y_int = torch::ones({4, 4}, at::kInt);
                (4, 4), dtype=torch.int32)           |
        """
    ),

    Setup.MESOSCALE: _setup_from_string(
        r"""
            Python                                   | C++
            ---------------------------------------- | ----------------------------------------
            x = torch.ones((4, 4))                   | auto x = torch::ones({4, 4});
            y = torch.ones((4, 4))                   | auto y = torch::ones({4, 4});
            bias = torch.ones((1,))                  | auto bias = torch::ones({1});
            eye_4 = torch.eye(4)                     | auto eye_4 = torch::eye(4);
        """
    ),

    Setup.AUTOGRAD: _setup_from_string(
        r"""
            Python                                   | C++
            ---------------------------------------- | ----------------------------------------
            # Inputs                                 | // Inputs
            x = torch.ones((1,))                     | auto x = torch::ones({1});
            y = torch.ones((1,))                     | auto y = torch::ones({1});
                                                     |
            # Weights                                | // Weights
            w0 = torch.ones(                         | auto w0 = torch::ones({1});
                (1,), requires_grad=True)            | w0.set_requires_grad(true);
            w1 = torch.ones(                         | auto w1 = torch::ones({1});
                (1,), requires_grad=True)            | w1.set_requires_grad(true);
            w2 = torch.ones(                         | auto w2 = torch::ones({2});
                (2,), requires_grad=True)            | w2.set_requires_grad(true);
        """
    ),

    Setup.EXAMPLE_FOR_ADHOC: {
        Language.PYTHON: r"x = torch.ones((1,))",
        Language.CPP: r"auto x = torch::ones({1});",
    },
}

# Ensure map is complete.
assert tuple(Setup) == tuple(SETUP_MAP.keys())
for k in Setup:
    assert tuple(SETUP_MAP[k].keys()) == (Language.PYTHON, Language.CPP)
