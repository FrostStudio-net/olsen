import pytest

from olsen.execution import KrakenPrivateAdapter, deterministic_client_order_id


def test_future_order_ids_are_deterministic_and_live_placement_is_disabled():
    first = deterministic_client_order_id("XBT/EUR", 1_700_000_000, "buy", "v0.2")
    second = deterministic_client_order_id("XBT/EUR", 1_700_000_000, "buy", "v0.2")
    adapter = KrakenPrivateAdapter()

    assert first == second
    assert adapter.reconcile(first, [{"client_order_id": first, "status": "open"}])["status"] == "open"
    with pytest.raises(RuntimeError, match="disabled"):
        adapter.place_order()
