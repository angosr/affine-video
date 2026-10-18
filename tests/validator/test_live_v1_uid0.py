import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

path = Path(__file__).resolve().parents[2] / "evidence/system-panel/operations/live_v1_uid0.py"
spec = importlib.util.spec_from_file_location("live_v1_uid0", path)
maintenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maintenance)


class LegacyMaintenanceTests(IsolatedAsyncioTestCase):
    async def exercise(self, *, heights, uid=0, matches=(), read_only=False):
        track = SimpleNamespace(
            digest=maintenance.LEGACY_DIGEST, track_id=maintenance.LEGACY_TRACK_ID,
            netuid=120, mechid=1, track_start_block=9310000,
            initial_champion=SimpleNamespace(hotkey="genesis"),
        )
        plan = SimpleNamespace(uids=(0,), weights=(1.0,), hotkeys=("genesis",))
        weights = SimpleNamespace(track_spec_digest=maintenance.LEGACY_DIGEST, plan=plan)
        reports, writes, closed, built = [], [], [], []
        pending = iter(heights)

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                closed.append(True)

            async def finalized_block(self):
                value = next(pending)
                if isinstance(value, Exception):
                    raise value
                return value

            def __getattr__(self, name):
                raise AssertionError(f"must not scan commitments or models: {name}")

        class Publisher:
            async def load(self, candidate):
                assert candidate is track
                return SimpleNamespace(matches=matches)

            async def publish(self, artifacts):
                writes.append(artifacts)

        async def read_state(client, *, netuid, finalized_block):
            assert netuid == 120
            return SimpleNamespace(uid_by_hotkey={"genesis": uid}, finalized_block=finalized_block)

        def build(candidate, chain, *, version_key):
            assert candidate is track and version_key == 0
            built.append(chain.finalized_block)
            return SimpleNamespace(weights=b"existing legacy weights", matches=())

        with (
            patch.object(maintenance, "read_weight_chain_state", read_state),
            patch.object(maintenance, "build_genesis_artifacts", build),
            patch.object(maintenance, "decode_weight_artifact", lambda _payload: weights),
        ):
            await maintenance.run_loop(
                track, Publisher(), client_factory=Client, report=reports.append,
                poll_seconds=0, maximum_attempts=len(heights), read_only=read_only,
            )
        return reports, writes, closed, built

    async def test_burn_continues_before_at_and_after_original_start_without_scanning(self):
        reports, writes, closed, built = await self.exercise(heights=[9309999, 9310000, 9310001])
        self.assertEqual(len(writes), 3)
        self.assertEqual(len(closed), 3)
        self.assertEqual(built, [9309999, 9310000, 9310001])
        self.assertEqual([r["after_original_start"] for r in reports], [False, True, True])

    async def test_chain_failure_recovers_in_same_process(self):
        reports, writes, closed, built = await self.exercise(
            heights=[OSError("head down"), 9310001],
        )
        self.assertEqual([r["event"] for r in reports], ["V1_UID0_RETRY", "V1_UID0_ONLY"])
        self.assertEqual(len(writes), 1)
        self.assertEqual(len(closed), 2)
        self.assertEqual(built, [9310001])

    async def test_reassigned_uid_or_existing_matches_fail_closed(self):
        for arguments in ({"uid": 1}, {"matches": ("an immutable match",)}):
            reports, writes, _closed, built = await self.exercise(heights=[9310001], **arguments)
            self.assertEqual(writes, [])
            self.assertEqual(built, [])
            self.assertEqual(reports[0]["event"], "V1_UID0_RETRY")

    async def test_read_only_probe_never_publishes(self):
        reports, writes, _closed, built = await self.exercise(heights=[9310001], read_only=True)
        self.assertEqual(writes, [])
        self.assertEqual(built, [9310001])
        self.assertFalse(reports[0]["public_write"])
