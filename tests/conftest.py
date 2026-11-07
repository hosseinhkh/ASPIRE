"""Shared pytest configuration.

The suite spans optional extras: the simulator venvs (``.venv-libero``,
``.venv-robosuite``) do not install the ``yam-real`` extra, but many YAM tests
build the synthetic station through fixtures that import it lazily. Those tests
used to show up there as dozens of errors that said nothing about the code under
test. A test that fails *only* because one of these packages is absent is
reported as skipped instead, with the extra named in the reason. With the extra
installed (``.venv-dev``, ``bootstrap_yam_real.sh``) nothing changes.
"""

from __future__ import annotations

import pytest

OPTIONAL_EXTRA_MODULES = {
    "mink": "yam-real",
    "portal": "yam-real",
}


def _missing_optional_module(error: BaseException | None) -> str | None:
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ModuleNotFoundError) and error.name in OPTIONAL_EXTRA_MODULES:
            return error.name
        error = error.__cause__ or error.__context__
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    outcome = yield
    report = outcome.get_result()
    if not report.failed or call.excinfo is None:
        return
    module = _missing_optional_module(call.excinfo.value)
    if module is None:
        return
    extra = OPTIONAL_EXTRA_MODULES[module]
    report.outcome = "skipped"
    report.longrepr = (
        str(item.path),
        item.location[1] or 0,
        f"Skipped: requires the `{extra}` extra ({module!r} is not installed)",
    )
