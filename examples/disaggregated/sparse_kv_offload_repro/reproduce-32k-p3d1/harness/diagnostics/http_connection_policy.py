# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use fresh worker HTTP connections for long-prompt benchmark bursts."""

import functools
import inspect
import json
import os
from pathlib import Path


def install(module):
    original = module.OpenAIHttpClient.__init__
    signature = inspect.signature(original)

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if bound.arguments["session"] is None:
            bound.arguments["session"] = module.aiohttp.ClientSession(
                connector=module.aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=True),
                timeout=module.aiohttp.ClientTimeout(total=bound.arguments["timeout_secs"]),
                max_field_size=module._PERF_METRICS_HEADER_BUDGET_BYTES,
            )
        original(*bound.args, **bound.kwargs)
        role = self._role.name
        record = {
            "role": role,
            "force_close": self._session.connector.force_close,
            "retry_disabled": self._no_retry,
            "max_retries": self._max_retries,
        }
        destination = os.environ.get("SWEEP_HTTP_POLICY_DIR")
        if destination:
            (Path(destination) / f"http-policy-{role}-pid{os.getpid()}.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
        print("SWEEP_HTTP_POLICY " + json.dumps(record), flush=True)

    module.OpenAIHttpClient.__init__ = initialize
