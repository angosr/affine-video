"""One-shot Bittensor commitment submission."""

from __future__ import annotations

from dataclasses import dataclass

import bittensor as bt

from miner.submit import ModelSubmission, PrivateModelSubmission, UploadActivation

_RAW_LIMIT = 128


class ChainSubmissionError(RuntimeError):
    """Raised when the chain rejects or cannot finalize a submission."""


@dataclass(frozen=True, slots=True)
class SubmissionReceipt:
    block_hash: str
    extrinsic_id: str
    message: str


def commitment_info(payload: bytes) -> dict[str, list[dict[str, bytes]]]:
    """Represent bytes as the exact runtime ``Data::Raw`` variant.

    Runtime metadata exposes a distinct ``RawN`` variant for every byte length,
    so the variant name must include the payload length.
    """

    if not payload:
        raise ValueError("commitment payload cannot be empty")
    if len(payload) > _RAW_LIMIT:
        raise ValueError(f"commitment payload exceeds {_RAW_LIMIT} bytes")
    return {"fields": [{f"Raw{len(payload)}": payload}]}


async def submit_model(
    submission: ModelSubmission | PrivateModelSubmission | UploadActivation,
    *,
    netuid: int,
    network: str,
    wallet_name: str,
    hotkey_name: str,
    wallet_path: str | None = None,
) -> SubmissionReceipt:
    """Submit one model reference, wait for finalization, and return."""

    if netuid < 0 or netuid > 0xFFFF:
        raise ValueError("netuid must fit in an unsigned 16-bit integer")
    if not network:
        raise ValueError("network cannot be empty")
    if not isinstance(submission, (ModelSubmission, PrivateModelSubmission, UploadActivation)):
        raise ValueError("submission must be an explicit HF or private model reference")
    if isinstance(submission, (PrivateModelSubmission, UploadActivation)) and netuid != 120:
        raise ValueError("private submission is restricted to SN120")
    if isinstance(submission, PrivateModelSubmission) and submission.manifest_sha256 is None:
        raise ValueError("private ready requires the signed manifest SHA-256")

    wallet_arguments = {"name": wallet_name, "hotkey": hotkey_name}
    if wallet_path is not None:
        wallet_arguments["path"] = wallet_path

    try:
        wallet = bt.Wallet(**wallet_arguments)
    except Exception as exc:
        raise ChainSubmissionError("the original wallet could not be opened") from exc
    if (isinstance(submission, (PrivateModelSubmission, UploadActivation))
            and getattr(wallet.hotkey, "crypto_type", None) != 0):
        raise ValueError("private registration and ready require an existing Ed25519 hotkey")
    try:
        call = bt.calls.Commitments.set_commitment(
            netuid=netuid,
            info=commitment_info(submission.encode()),
        )
        async with bt.Client(network) as client:
            result = await client.submit_call(
                call,
                wallet,
                signer="hotkey",
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
    except Exception as exc:
        raise ChainSubmissionError("commitment transaction did not complete") from exc

    if not result.success:
        raise ChainSubmissionError(result.message or "commitment transaction failed")
    if not result.block_hash or not result.extrinsic_id:
        raise ChainSubmissionError("finalized commitment transaction is missing chain identifiers")
    return SubmissionReceipt(
        block_hash=result.block_hash,
        extrinsic_id=result.extrinsic_id,
        message=result.message,
    )
