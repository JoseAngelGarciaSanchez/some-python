import functools
import platform
import sys
import sysconfig

from labs.lab5_freethreading import RuntimeInfo, Timing, measure, pure_python_task, run_in_interpreters, runtime_info


def test_runtime_info_describes_this_interpreter():
    info = runtime_info()
    assert isinstance(info, RuntimeInfo)
    assert info.version == platform.python_version()
    assert info.freethreaded is bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    assert info.gil_enabled is sys._is_gil_enabled()
    assert info.cpus >= 1


def test_measure_returns_a_timing():
    timing = measure(functools.partial(pure_python_task, 20_000), n_tasks=2)
    assert isinstance(timing, Timing)
    assert timing.serial_s > 0 and timing.threaded_s > 0


def test_measure_warms_up_then_runs_each_batch():
    calls = []
    measure(lambda: calls.append(None), n_tasks=3)
    assert len(calls) == 1 + 3 + 3, "expected 1 warm-up, 3 serial and 3 threaded calls"


def test_run_in_interpreters():
    assert run_in_interpreters(pure_python_task, [5_000, 6_000], workers=2) == [669, 783]
