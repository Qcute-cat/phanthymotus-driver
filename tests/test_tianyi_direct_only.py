"""Direct ROS callback dispatch without changing bc14660 admission/lease rules."""

import importlib.util
import json
from pathlib import Path
import threading
import time
import types

import pytest

from common.control import ControlSink
from test_tianyi_loco_servo import command, setup, start
from test_tianyi_sdk_worker import native_worker


def message(cmd):
    return types.SimpleNamespace(data=json.dumps(cmd))


def test_callback_checks_and_writes_synchronously_on_the_calling_thread(setup, monkeypatch):
    plugin = start(setup)
    events = []
    callback_thread = threading.get_ident()
    submit, write = plugin._sink.submit, plugin._backend.write

    def checked(cmd):
        events.append(("sink-enter", threading.get_ident()))
        result = submit(cmd)
        events.append(("sink-return", threading.get_ident()))
        return result

    def written(values, deadline, **kwargs):
        events.append(("sdk-write", threading.get_ident()))
        return write(values, deadline, **kwargs)

    monkeypatch.setattr(plugin._sink, "submit", checked)
    monkeypatch.setattr(plugin._backend, "write", written)
    plugin._on_message(message(command(setup)))
    events.append(("callback-return", threading.get_ident()))
    assert events == [(name, callback_thread) for name in
                      ("sink-enter", "sdk-write", "sink-return", "callback-return")]
    assert plugin._counts["accepted"] == plugin._counts["published"] == 1
    assert plugin._backend.messages[-1] == pytest.approx([0.03, 0, 0, 0, 0, 0.06])


def test_continuous_10hz_callbacks_never_insert_zero_or_wait_for_a_tick(setup):
    plugin = start(setup)
    for seq in range(1, 21):
        plugin._on_message(message(command(setup, seq=seq)))
        assert plugin._counts["accepted"] == seq
        assert len(plugin._backend.messages) == seq
        assert plugin._backend.messages[-1][0] > 0
        assert plugin._backend.messages[-1][5] > 0
        setup.clock.advance()
        plugin._tick()
    assert plugin._counts["holds"] == plugin._counts["dropped"] == 0
    assert plugin._backend.messages[-1] == pytest.approx([0.2, 0, 0, 0, 0, 0.4])


def test_watchdog_is_one_ros_timer_without_a_python_dispatch_thread(setup, monkeypatch):
    def forbidden_thread(*args, **kwargs):
        raise AssertionError("loco_servo must use its input callback, not a Python dispatch thread")

    monkeypatch.setattr(setup.module.threading, "Thread", forbidden_thread)
    plugin = start(setup)
    setup.launch_watchdog(plugin)
    node = setup.ros.executor_core.nodes[0]
    backend = plugin._backend
    assert len(node.timers) == 1
    assert node.timers[0][0] == pytest.approx(0.02)
    assert not node.guards
    assert plugin._info()["dispatch"]["mode"] == "direct_callback"
    assert plugin._info()["dispatch"]["watchdog_active"] is True
    assert plugin.dispatch("start", {})["state"] == "running"
    assert len(node.timers) == 1
    plugin._on_message(message(command(setup)))
    setup.clock.advance(0.21)
    node.timers[0][1]()
    assert backend.messages[-1] == [0] * 6
    assert plugin.stop()["state"] == "idle"
    assert not node.timers and node.destroyed
    assert not backend.connected and not setup.ros.executor_core.nodes
    assert plugin._watchdog_timer is None


def test_watchdog_timer_creation_failure_cleans_up_nodes_and_backend(setup, monkeypatch):
    backends, nodes = [], []
    backend_class = setup.module.SlamtecVelocityBackend

    def backend_factory(config):
        backend = backend_class(config)
        backends.append(backend)
        return backend

    def fail_timer(node, period, callback):
        nodes.append(node)
        raise RuntimeError("timer unavailable")

    monkeypatch.setattr(setup.module, "SlamtecVelocityBackend", backend_factory)
    monkeypatch.setattr(__import__("rclpy.node", fromlist=["Node"]).Node,
                        "create_timer", fail_timer)
    plugin = setup.create()
    monkeypatch.setattr(plugin, "_start_watchdog", lambda generation: setup.launch_watchdog(plugin))
    result = plugin.dispatch("start", {})
    assert result == {"state": "error", "error": "watchdog start failed: timer unavailable"}
    assert nodes[0].destroyed and not nodes[0].timers
    assert backends[0].connected is False
    assert not plugin._running and plugin._sink is None and plugin._backend is None
    assert not setup.ros.executor_core.nodes


def test_watchdog_exception_stops_and_disallows_resume_until_restart(setup, monkeypatch):
    plugin = start(setup)
    setup.launch_watchdog(plugin)
    plugin._on_message(message(command(setup)))

    def broken_tick():
        raise RuntimeError("watchdog tick failed")

    monkeypatch.setattr(plugin._sink, "tick", broken_tick)
    setup.ros.executor_core.nodes[0].timers[0][1]()
    assert plugin._backend.messages[-1] == [0] * 6
    assert plugin._paused and plugin._watchdog_failed
    assert plugin._last == {"verdict": "ERROR", "reason": "watchdog failed: watchdog tick failed"}
    assert plugin.dispatch("resume", {}) == {
        "state": "error", "error": "watchdog failed; stop/start the card"}


def test_timer_teardown_failure_does_not_leave_the_backend_or_node_active(setup, monkeypatch):
    plugin = start(setup)
    setup.launch_watchdog(plugin)
    plugin._on_message(message(command(setup)))
    node, backend = setup.ros.executor_core.nodes[0], plugin._backend

    def fail_destroy(timer):
        raise RuntimeError("timer teardown unavailable")

    monkeypatch.setattr(node, "destroy_timer", fail_destroy)
    assert plugin.stop()["state"] == "idle"
    assert plugin._watchdog_timer is None and plugin._backend is None
    assert not backend.connected and backend.messages[-1] == [0] * 6
    assert node.destroyed and not setup.ros.executor_core.nodes


def test_apply_start_rate_is_bounded_at_20hz_but_zero_bypasses_it(setup, monkeypatch):
    plugin = start(setup)
    original_write = plugin._backend.write

    def short_sdk_call(values, deadline, **kwargs):
        setup.clock.advance(0.01)
        return original_write(values, deadline, **kwargs)

    monkeypatch.setattr(plugin._backend, "write", short_sdk_call)
    plugin._on_message(message(command(setup, seq=1)))
    setup.clock.advance(0.03)  # 40ms since the previous SDK call began.
    plugin._on_message(message(command(setup, seq=2)))
    assert plugin._last == {"verdict": "DROPPED", "reason": "command rate exceeds 20 Hz"}
    assert len(plugin._backend.messages) == 1
    setup.clock.advance(0.02)  # 60ms since that call began: now admit it.
    plugin._on_message(message(command(setup, seq=3)))
    assert len(plugin._backend.messages) == 2
    plugin._on_message(message(command(setup, seq=4, values=[0] * 6)))
    assert plugin._backend.messages[-1] == [0] * 6
    assert plugin._counts["accepted"] == 3


@pytest.mark.parametrize("extra,remaining", [
    ({}, 200),
    ({"ttl_ms": 1000}, 300),
    ({"ttl_ms": 1000, "obs_age": 450}, 50),
])
def test_native_deadline_remains_minimum_of_original_ttl_observation_and_watchdog(setup, extra, remaining):
    plugin = start(setup)
    extra = dict(extra)
    stamp = int(setup.clock.time() * 1000)
    if "obs_age" in extra:
        extra["obs_stamp_ms"] = stamp - extra.pop("obs_age")
    plugin._on_message(message(command(setup, **extra)))
    assert plugin._backend.deadlines == [stamp + remaining]
    assert plugin._info()["control_interface"]["rate"] == {
        "max_hz": 20, "expected_hz": 10.0, "watchdog_ms": 300, "max_obs_age_ms": 500}


@pytest.mark.parametrize("kind", ["ttl", "observation", "sequence"])
def test_stale_ingress_does_not_write_or_refresh_the_last_valid_frame(setup, kind):
    plugin = start(setup)
    original = command(setup, seq=1)
    plugin._on_message(message(original))
    valid_at, expiry = plugin._last_valid_at, plugin._lease_expiry_ms
    setup.clock.advance(0.1)
    stale = command(setup, seq=2)
    if kind == "ttl":
        stale["ttl_ms"] = 20
        stale["stamp_ms"] -= 30
    elif kind == "observation":
        stale["obs_stamp_ms"] -= 501
    else:
        stale = original
    plugin._on_message(message(stale))
    assert plugin._last["verdict"] == "DROPPED"
    assert plugin._counts["accepted"] == 1
    assert len(plugin._backend.messages) == 1
    assert plugin._last_valid_at == valid_at and plugin._lease_expiry_ms == expiry


def test_old_subscription_and_timer_generations_cannot_write_to_restarted_card(setup):
    plugin = start(setup)
    setup.launch_watchdog(plugin)
    old_node = setup.ros.executor_core.nodes[0]
    old_subscription, old_timer = old_node.subscriptions[0][2], old_node.timers[0][1]
    assert plugin.stop()["state"] == "idle"
    assert plugin.dispatch("start", {})["state"] == "running"
    setup.clock.advance()
    plugin._on_message(message(command(setup)))
    backend = plugin._backend
    before = list(backend.messages)
    setup.clock.advance(0.31)
    old_subscription(message(command(setup, seq=2)))
    old_timer()
    assert backend.messages == before
    assert plugin._counts["accepted"] == 1


def test_explicit_zero_pause_and_stop_keep_immediate_stopping_semantics(setup):
    plugin = start(setup)
    plugin._on_message(message(command(setup)))
    # Zero must bypass both rate and acceleration limiting, even in the same millisecond.
    plugin._on_message(message(command(setup, values=[0] * 6, seq=2)))
    assert plugin._backend.messages[-1] == [0] * 6
    setup.clock.advance()
    plugin._on_message(message(command(setup, seq=3)))
    backend = plugin._backend
    assert backend.messages[-1][0] > 0
    assert plugin.dispatch("pause", {})["state"] == "paused"
    assert backend.messages[-1] == [0] * 6
    written = len(backend.messages)
    setup.clock.advance()
    plugin._on_message(message(command(setup, seq=4)))
    assert len(backend.messages) == written
    assert plugin.stop()["state"] == "idle"
    plugin._on_message(message(command(setup, seq=5)))
    assert not backend.connected and plugin._backend is None


def test_direct_callback_full_native_sdk_chain_retains_action_for_a_10hz_stream(
        setup, monkeypatch, native_worker):
    driver = Path(__file__).resolve().parents[1] / "x-humanoid/tianyi2.0"
    spec = importlib.util.spec_from_file_location("tianyi_direct_only_real_backend", driver / "slamtec_velocity.py")
    backend_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend_module)
    # Both plugin and the actual sink use wall/monotonic real clocks. Fake-clock
    # epochs would make the native worker's real command-deadline check meaningless.
    monkeypatch.setattr(setup.module, "time", time)
    monkeypatch.setattr(setup.module, "ControlSink", ControlSink)
    monkeypatch.setattr(setup.module, "SlamtecVelocityBackend", backend_module.SlamtecVelocityBackend)
    monkeypatch.setattr(setup.module, "CommandLeaseExpired", backend_module.CommandLeaseExpired)
    plugin = setup.create(sdk_binary=str(native_worker), sdk_host="ok")
    assert plugin.dispatch("start", {})["state"] == "running"
    time.sleep(0.005)
    action_id = None
    started = time.monotonic()
    for seq in range(1, 7):
        delay = started + (seq - 1) * 0.1 - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        stamp = int(time.time() * 1000)
        cmd = {"schema": "motus.control/1", "mode": "twist", "dof": 6,
               "values": [0.15, 0, 0, 0, 0, -0.3], "seq": seq,
               "source": "navi", "priority": 0, "stamp_ms": stamp,
               "obs_stamp_ms": stamp, "ttl_ms": 200}
        plugin._on_message(message(cmd))
        info = plugin._info()
        assert info["counts"]["accepted"] == seq
        assert info["last"]["output"][0] > 0 and info["last"]["output"][5] < 0
        backend = info["backend"]
        action_id = action_id or backend["action_id"]
        assert backend["action_id"] == action_id > 0
        stats = backend["native"]["stats"]
        assert stats["action_starts"] == 1 and stats["set_attempts"] == seq
        assert stats["stop_attempts"] == stats["confirmed_stops"] == 0
        assert 0 < stats["last_set"]["remaining_after_set_ms"] <= 200
        plugin._tick()
    backend = plugin._backend
    assert plugin.dispatch("pause", {})["state"] == "paused"
    stopped = backend.info()
    assert stopped["action_id"] == 0
    assert stopped["native"]["stats"]["confirmed_stops"] == 1
    assert plugin.stop()["state"] == "idle"
    assert backend.info()["connected"] is False
