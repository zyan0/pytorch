# gh/taylorrobie/callgrind_ci_prototype

from frontend.interface import ab_test


def main() -> None:
    ab_test(
        source_a="source activate throwaway",
        source_b="source activate test_env",
        ad_hoc=False,
    )


if __name__ == "__main__":
    main()
