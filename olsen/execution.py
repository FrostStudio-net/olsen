import hashlib
from collections.abc import Iterable, Mapping


def deterministic_client_order_id(pair: str, candle_timestamp: int, side: str, strategy: str) -> str:
    """Stable identifier reserved for future reconciled order submission."""
    payload = f"{strategy}|{pair}|{candle_timestamp}|{side}".encode()
    return f"olsen-{hashlib.sha256(payload).hexdigest()[:24]}"


class KrakenPrivateAdapter:
    """Disabled placeholder: v0.2 cannot place live orders."""

    live_execution_enabled = False

    def place_order(self, *args, **kwargs):
        raise RuntimeError("Live order placement is disabled in Olsen v0.2.")

    def reconcile(
        self, client_order_id: str, orders: Iterable[Mapping[str, object]]
    ) -> Mapping[str, object] | None:
        """Reconcile a future client ID against already-fetched order records.

        This is deliberately a pure function over caller-supplied records; v0.2 has
        no authenticated transport and cannot fetch or submit private orders.
        """
        matches = [order for order in orders if order.get("client_order_id") == client_order_id]
        if len(matches) > 1:
            raise RuntimeError(f"Duplicate reconciled orders for {client_order_id}")
        return matches[0] if matches else None
