"""Frozen GPU report summaries, not protocol identity or candidate authorization.

Historical records retain the measured b81/860 identities. A later scoring runtime
must produce its own evidence before it can attach records. Published sequences
and recorded identities are immutable: append rather than renumber. No private
source identities, inputs, paths, or credentials belong in this table.
"""

_HISTORICAL_RUNTIME = {
    "track_id": "veda-sn120-v2",
    "netuid": 120,
    "track_spec_digest": "8608cf5dcb4e2ab68933950a333dbf95184732b93c8a1ba9332f3a0b90af7c34",
    "track_contract_sha256": "2c1634e3eee6d970d432c31d71c4c5ac4a983d8fc95872cb251c297e8606f95f",
    "runtime_contract_sha256": "92dfa83a26292df75e8bfd329e972eb1a92002884596032887fe4aac54a09021",
    "source_sha256": "3fdf7148a4da81119ee51b6557e2a08066c4ffdc69bdb759fd567e524ab0c025",
    "environment_sha256": "cde5278e70b6e5f61df2b9c2bd8810de5584579a4fc3f59e062cc7d3c417a124",
    "image_sha256": "3749e5b063b50d66c4d676dd2c79c10c460e5d7f1de9e7713f7444428bcc2c2c",
    "hardware": "NVIDIA H200",
    "scope": "isolated_accelerated_chain_fixture",
    "confidence": 0.9975,
    "minimum_gain": 0.005,
    "budgets": (16, 32, 64),
    "cluster_unit": "source_video",
}

_BASELINE = {
    "model_id": "veda_baseline",
    "label": "VEDA baseline",
    "sha256": "64addb5da430059e37e405d28ca0f2887132905f077ae92fba86982c8b0e9d3f",
    "model_digest": "3fef3eca0eebee238b1ed9c692b217dd25cb1082ef94779ec2ef7c93ae0560c9",
}
_SHUFFLED = {
    "model_id": "shuffled_heads_control",
    "label": "Shuffled-heads control",
    "sha256": "d9e549ef355ae494947a47b4891c92716b85e75a82fb868d8277db1e090017ec",
    "model_digest": "7c7798ca66eed2a9a8a48bbc5091d361c2ba9b56a908aa790a33c1434b92a42b",
}
_EXACT_COHORT = {
    "cohort": "audit-9311300",
    "partition": "audit",
    "generation_block": 9311300,
    "plan_digest": "8a64319c80c15d946d1aceee835350def0500a0d9c1a392516aaf8c08c1fc973",
    "cache_sha256": "25394b9282a9b79fd9f7b589281d49d17abafb753e69548db5af2124d7a8acb3",
    "ready_sha256": "daa9e35fdababa0ea5f74d23bc19c6beac60e3316da759978b8f48b1f9cdbce6",
    "evidence_sha256": "4a2f9c84bef9067522a83d80b8674d2eba7438439aa8495cc0c550852fa66809",
    "cases": 384,
    "independent_videos": 16,
    "independent_videos_per_stratum": 4,
}
_CONFIRMATION_COHORT = {
    "cohort": "audit-9312500",
    "partition": "audit",
    "generation_block": 9312500,
    "plan_digest": "3a508e53304f248015cca3f68821b47dcdf05c892ded6fe40b56c6bc48538b87",
    "cache_sha256": "38394c068b9f540c7617d672ffb67bd33ee9ac44fe63d9c39518b3c6500a3fbb",
    "ready_sha256": "b077d148244c5b91656684d71add786ac8f26660d69963d646fddefecbafea13",
    "evidence_sha256": "59b6aba46debcb3001848ca667443106b25fc075bc7071acf0487ceb5d37057e",
    "cases": 48,
    "independent_videos": 16,
    "independent_videos_per_stratum": 4,
}

BENCHMARKS = (
    {
        **_HISTORICAL_RUNTIME,
        "sequence": 1,
        "phase": "exact",
        "reference": _BASELINE,
        "challenger": _BASELINE,
        "batches": (
            {
                **_EXACT_COHORT,
                "reference_score": 0.9491625628902893,
                "challenger_score": 0.9491625628902893,
                "challenger_gain_lower_bound": 0.0,
                "reference_gain_lower_bound": 0.0,
            },
        ),
    },
    {
        **_HISTORICAL_RUNTIME,
        "sequence": 2,
        "phase": "exact",
        "reference": _BASELINE,
        "challenger": _SHUFFLED,
        "batches": (
            {
                **_EXACT_COHORT,
                "reference_score": 0.9491625628902893,
                "challenger_score": 0.7885866430037899,
                "challenger_gain_lower_bound": -0.2561681043494346,
                "reference_gain_lower_bound": 0.09963065729838526,
            },
        ),
    },
    {
        **_HISTORICAL_RUNTIME,
        "sequence": 3,
        "phase": "full_video",
        "reference": _BASELINE,
        "challenger": _BASELINE,
        "batches": (
            {
                **_CONFIRMATION_COHORT,
                "reference_score": 0.7619235479211722,
                "challenger_score": 0.7619235479211722,
                "challenger_gain_lower_bound": 0.0,
                "reference_gain_lower_bound": 0.0,
            },
        ),
    },
    {
        **_HISTORICAL_RUNTIME,
        "sequence": 4,
        "phase": "full_video",
        "reference": _BASELINE,
        "challenger": _SHUFFLED,
        "batches": (
            {
                **_CONFIRMATION_COHORT,
                "reference_score": 0.7619235479211722,
                "challenger_score": 0.6570879920438147,
                "challenger_gain_lower_bound": -0.19131944739869802,
                "reference_gain_lower_bound": 0.04190771739740623,
            },
        ),
    },
)
