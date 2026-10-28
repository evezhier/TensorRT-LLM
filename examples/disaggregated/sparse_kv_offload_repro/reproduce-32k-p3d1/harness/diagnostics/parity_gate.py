# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in receive-publication barrier for capacity and matched-batch parity probes."""

import functools
import json
import os
import time
from pathlib import Path


def install(module) -> None:
    """Instrument diagnostic servers; performance servers do not install this hook."""
    if module.__name__.endswith(".coordinator"):
        original_poll = module.DisaggTransferCoordinator.poll_gen_transfers

        @functools.wraps(original_poll)
        def poll(self):
            self._transceiver._sweep_symmetric_poll = True
            try:
                return original_poll(self)
            finally:
                self._transceiver._sweep_symmetric_poll = False

        module.DisaggTransferCoordinator.poll_gen_transfers = poll
        return
    control_path = Path(os.environ["LONG_PARITY_GATE_PATH"])
    original = module.KvCacheTransceiverV2.check_gen_transfer_status
    cached = {"epoch": "", "enabled": False}
    last_read = 0.0

    @functools.wraps(original)
    def gated(self, at_least_request_num):
        nonlocal cached, last_read
        now = time.monotonic()
        if now - last_read >= 0.2:
            try:
                cached = json.loads(control_path.read_text())
            except FileNotFoundError:
                cached = {"epoch": "", "enabled": False}
            last_read = now
        if not getattr(self, "_sweep_symmetric_poll", False):
            if cached.get("enabled"):
                return module.GenTransferStatus([], [], [])
            return original(self, at_least_request_num)
        completed, failed = self._collect_done(self._recv_sessions, self._recv_reqs)
        peers = self._dist.allgather((cached, completed, failed))
        active = max((entry[0] for entry in peers), key=lambda cfg: cfg["epoch"])
        epoch = active["epoch"]
        released = getattr(self, "_sweep_released_epoch", None)
        if not active.get("enabled") or released == epoch:
            return original(self, at_least_request_num)
        ready_ids = set(rid for _, ready, _ in peers for rid in ready)
        failed_ids = set(rid for _, _, errors in peers for rid in errors)
        state = getattr(self, "_sweep_capacity_state", None)
        if state is None or state[0] != epoch or state[1] != len(ready_ids):
            state = (epoch, len(ready_ids), now)
            self._sweep_capacity_state = state
        plateau_seconds = active.get("plateau_seconds", 0)
        local_plateau = bool(ready_ids and plateau_seconds and now - state[2] >= plateau_seconds)
        statuses = self._dist.allgather((local_plateau, time.time() > active["deadline"]))
        plateau = all(status[0] for status in statuses)
        expired = any(status[1] for status in statuses)
        target = active["concurrency"]
        if len(ready_ids) >= target or failed_ids or expired or plateau:
            self._sweep_released_epoch = epoch
            record = {
                "epoch": epoch,
                "target": target,
                "completed": len(ready_ids),
                "completed_per_rank": [len(ready) for _, ready, _ in peers],
                "failed": len(failed_ids),
                "timed_out": expired,
                "plateau": plateau,
                "passed": len(ready_ids) == target and not failed_ids and not expired,
            }
            path = control_path.parent / f"{epoch}-gate-rank{self._mapping.rank}.json"
            path.write_text(json.dumps(record, indent=2) + "\n")
            return original(self, at_least_request_num)
        return module.GenTransferStatus([], [], [])

    module.KvCacheTransceiverV2.check_gen_transfer_status = gated
