import textwrap

from torch.utils.benchmark import Language

from tasks.spec import Setup


_SetupMap = {
    Setup.NONE: {
        Language.PYTHON: "pass",
        Language.CPP: "",
    },

    Setup.GENERIC: {
        Language.PYTHON: r"""
            x = torch.ones((4, 4))
            y_float = torch.ones((4, 4))
            y_int = torch.ones((4, 4), dtype=torch.int32)
        """,
        Language.CPP: r"""
            auto x = torch::ones({4, 4});
            auto y_float = torch::ones({4, 4});
            auto y_int = torch::ones({4, 4}, at::kInt);
        """,
    },

    Setup.MESOSCALE: {
        Language.PYTHON: r"""
            x = torch.ones((4, 4))
            y = torch.ones((4, 4))
            bias = torch.ones((1,))
            eye_4 = torch.eye(4)
        """,
        Language.CPP: r"""
            auto x = torch::ones({4, 4});
            auto y = torch::ones({4, 4});
            auto bias = torch::ones({1});
            auto eye_4 = torch::eye(4);
        """,
    },

    Setup.AUTOGRAD: {
        Language.PYTHON: r"""
            x = torch.ones((1,))
            y = torch.ones((1,))
            w0 = torch.ones((1,), requires_grad=True)
            w1 = torch.ones((1,), requires_grad=True)
            w2 = torch.ones((2,), requires_grad=True)
        """,
        Language.CPP: r"""
            auto x = torch::ones({1});
            auto y = torch::ones({1});

            auto w0 = torch::ones({1});
            w0.set_requires_grad(true);

            auto w1 = torch::ones({1});
            w1.set_requires_grad(true);

            auto w2 = torch::ones({2});
            w2.set_requires_grad(true);
        """,
    },
}


assert tuple(Setup) == tuple(_SetupMap.keys())
for k in Setup:
    assert tuple(_SetupMap[k].keys()) == (Language.PYTHON, Language.CPP)

SetupMap = {
    k: {lang: textwrap.dedent(setup) for lang, setup in v.items()}
    for k, v in _SetupMap.items()
}
