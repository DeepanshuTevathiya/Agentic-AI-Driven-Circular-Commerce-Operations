"""
Lead 2 Shipment Tracking Contracts, Lifecycle & Deterministic Simulator
========================================================================
Phase 5.5.1 Implementation:
Defines typed, immutable tracking event contracts, operational shipment lifecycle
state machine, carrier provider abstraction, and a deterministic simulation provider.

ARCHITECTURAL PRINCIPLES:
1. STRICT VALIDATION & IMMUTABILITY:
   All tracking events and records are frozen Pydantic models with strict field,
   identifier, and ISO 8601 UTC timestamp validation.
2. CAUSAL ORDERING & TEMPORAL DISCRIMINATION:
   Distinguishes physical event occurrence time ('event_timestamp') from system
   ingestion time ('received_timestamp'). Enforces received_timestamp >= event_timestamp.
3. IDEMPOTENT INGESTION & CONFLICT DETECTION:
   Repeated provider events with identical canonical payload fingerprints are
   deduplicated as safe no-ops. Conflicting reuse of provider_event_id is rejected.
4. OUT-OF-ORDER TOLERANCE WITHOUT REGRESSION:
   Late-arriving tracking updates from prior transit legs are preserved in the
   chronological audit trail, but never regress terminal or advanced shipment statuses.
5. ISOLATION BETWEEN SHIPMENTS:
   Each shipment maintains completely isolated tracking state. Multiple shipments
   under a single FulfilmentOrder operate independently.
6. CRITICAL BUSINESS BOUNDARY:
   A carrier reporting 'DELIVERED' signifies physical transport completion only.
   Tracking NEVER creates GRNs, increases accepted quantities, modifies inventory
   batches, alters pipeline inventory, or modifies the approved PurchaseOrder.
7. DETERMINISTIC SIMULATION:
   Zero random numbers, zero real I/O, zero network calls. Deterministic timelines
   for normal delivery, delayed delivery, exception recovery, and return-to-origin.
"""

from abc import ABC, abstractmethod
from datetime import datetime as dt_datetime, timedelta, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schemas import (
    FulfilmentOrder,
    compute_payload_fingerprint,
    validate_and_normalize_iso_timestamp,
    validate_timestamp_sequence,
)


# -----------------------------------------------------------------------------
# Tracking Enums
# -----------------------------------------------------------------------------

class ShipmentTrackingStatus(str, Enum):
    """
    Operational lifecycle tracking statuses for an in-transit supplier shipment.
    """
    LABEL_CREATED = "LABEL_CREATED"
    PICKED_UP = "PICKED_UP"
    IN_TRANSIT = "IN_TRANSIT"
    AT_HUB = "AT_HUB"
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"
    DELIVERED = "DELIVERED"
    DELIVERY_EXCEPTION = "DELIVERY_EXCEPTION"
    RETURN_INITIATED = "RETURN_INITIATED"
    RETURNED = "RETURNED"


class TrackingScenario(str, Enum):
    """
    Deterministic simulation scenarios supported by SimulatedTrackingProvider.
    """
    NORMAL_DELIVERY = "NORMAL_DELIVERY"
    DELAYED_DELIVERY = "DELAYED_DELIVERY"
    DELIVERY_EXCEPTION_AND_RECOVERY = "DELIVERY_EXCEPTION_AND_RECOVERY"
    RETURN_TO_ORIGIN = "RETURN_TO_ORIGIN"
    CUSTOM = "CUSTOM"


# -----------------------------------------------------------------------------
# Tracking Exceptions
# -----------------------------------------------------------------------------

class TrackingError(Exception):
    """Base exception for shipment tracking domain errors."""
    pass


class InvalidTrackingEventError(TrackingError, ValueError):
    """Raised when a tracking event payload fails schema, field, or timestamp validation."""
    pass


class InvalidTrackingTransitionError(TrackingError, ValueError):
    """Raised when an event attempts an illegal lifecycle transition or regresses a terminal state."""
    pass


class TrackingIdempotencyConflictError(TrackingError, ValueError):
    """Raised when a provider_event_id is reused with differing payload or attributes."""
    pass


class ShipmentNotFoundError(TrackingError, KeyError):
    """Raised when a tracking query references an unregistered or unknown shipment ID."""
    pass


# -----------------------------------------------------------------------------
# Lifecycle Transition Matrix
# -----------------------------------------------------------------------------

VALID_TRACKING_TRANSITIONS: Dict[ShipmentTrackingStatus, Tuple[ShipmentTrackingStatus, ...]] = {
    ShipmentTrackingStatus.LABEL_CREATED: (
        ShipmentTrackingStatus.PICKED_UP,
        ShipmentTrackingStatus.IN_TRANSIT,
        ShipmentTrackingStatus.AT_HUB,
        ShipmentTrackingStatus.DELIVERY_EXCEPTION,
        ShipmentTrackingStatus.RETURN_INITIATED,
    ),
    ShipmentTrackingStatus.PICKED_UP: (
        ShipmentTrackingStatus.IN_TRANSIT,
        ShipmentTrackingStatus.AT_HUB,
        ShipmentTrackingStatus.OUT_FOR_DELIVERY,
        ShipmentTrackingStatus.DELIVERY_EXCEPTION,
        ShipmentTrackingStatus.RETURN_INITIATED,
    ),
    ShipmentTrackingStatus.IN_TRANSIT: (
        ShipmentTrackingStatus.AT_HUB,
        ShipmentTrackingStatus.OUT_FOR_DELIVERY,
        ShipmentTrackingStatus.DELIVERED,
        ShipmentTrackingStatus.DELIVERY_EXCEPTION,
        ShipmentTrackingStatus.RETURN_INITIATED,
    ),
    ShipmentTrackingStatus.AT_HUB: (
        ShipmentTrackingStatus.IN_TRANSIT,        # Transshipment / line-haul to another hub
        ShipmentTrackingStatus.OUT_FOR_DELIVERY,  # Final delivery run dispatch
        ShipmentTrackingStatus.DELIVERED,
        ShipmentTrackingStatus.DELIVERY_EXCEPTION,
        ShipmentTrackingStatus.RETURN_INITIATED,
        ShipmentTrackingStatus.RETURNED,          # Handed back to supplier at vendor return hub
    ),
    ShipmentTrackingStatus.OUT_FOR_DELIVERY: (
        ShipmentTrackingStatus.DELIVERED,
        ShipmentTrackingStatus.DELIVERY_EXCEPTION, # Bay closed, vehicle breakdown, weather
        ShipmentTrackingStatus.AT_HUB,             # End of day return to hub for re-dispatch
        ShipmentTrackingStatus.RETURN_INITIATED,   # Immediate refusal / RTO
    ),
    ShipmentTrackingStatus.DELIVERY_EXCEPTION: (
        ShipmentTrackingStatus.OUT_FOR_DELIVERY,  # Delivery re-attempt
        ShipmentTrackingStatus.IN_TRANSIT,        # Resumed transit / rerouting
        ShipmentTrackingStatus.AT_HUB,            # Holding at sorting hub
        ShipmentTrackingStatus.DELIVERED,         # Resolved on-site and delivered
        ShipmentTrackingStatus.RETURN_INITIATED,  # Refused or failed repeatedly -> RTO
    ),
    ShipmentTrackingStatus.RETURN_INITIATED: (
        ShipmentTrackingStatus.IN_TRANSIT,        # Transit back to supplier
        ShipmentTrackingStatus.AT_HUB,            # Return sorting facility
        ShipmentTrackingStatus.RETURNED,          # Handed back to supplier warehouse
        ShipmentTrackingStatus.DELIVERY_EXCEPTION,# Exception during return transit
    ),
    ShipmentTrackingStatus.DELIVERED: (),   # Terminal state
    ShipmentTrackingStatus.RETURNED: (),    # Terminal state
}

TERMINAL_TRACKING_STATUSES: Tuple[ShipmentTrackingStatus, ...] = (
    ShipmentTrackingStatus.DELIVERED,
    ShipmentTrackingStatus.RETURNED,
)


def is_valid_tracking_transition(
    current_status: Optional[ShipmentTrackingStatus],
    new_status: ShipmentTrackingStatus,
) -> bool:
    """
    Validates whether a transition from current_status to new_status is permitted.

    Rules:
    1. If current_status is None, any non-terminal initial status is permitted.
    2. Terminal statuses (DELIVERED, RETURNED) can never transition to any status.
    3. Checkpoints with the same status (e.g. IN_TRANSIT -> IN_TRANSIT, AT_HUB -> AT_HUB,
       DELIVERY_EXCEPTION -> DELIVERY_EXCEPTION) are permitted to record intermediate updates.
    4. Otherwise, new_status must be in VALID_TRACKING_TRANSITIONS[current_status].
    """
    if current_status is None:
        return new_status not in TERMINAL_TRACKING_STATUSES

    if current_status in TERMINAL_TRACKING_STATUSES:
        return False

    if current_status == new_status:
        # Same-status progression (intermediate checkpoints) is permitted for non-terminal states
        return True

    return new_status in VALID_TRACKING_TRANSITIONS.get(current_status, ())


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

def _validate_non_empty_str(v: str, field_name: str) -> str:
    """Validates that a string is non-empty and non-whitespace."""
    if not isinstance(v, str) or not v.strip():
        raise InvalidTrackingEventError(f"Field '{field_name}' must be a non-empty string.")
    return v.strip()


class TrackingLocation(BaseModel):
    """
    Structured location of a shipment tracking checkpoint.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    hub_id: Optional[str] = Field(default=None, description="Hub, sorting center, or facility identifier")
    facility_name: Optional[str] = Field(default=None, description="Human-readable facility or bay name")
    city: Optional[str] = Field(default=None, description="City or municipality")
    state: Optional[str] = Field(default=None, description="State or province")
    postal_code: Optional[str] = Field(default=None, description="Postal code or PIN")

    @field_validator("hub_id", "facility_name", "city", "state", "postal_code")
    @classmethod
    def validate_strings(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            raise InvalidTrackingEventError(f"Field '{info.field_name}', if provided, cannot be empty or whitespace.")
        return clean

    @property
    def display_name(self) -> str:
        """Compact string representation for audit logs."""
        parts = [p for p in [self.facility_name or self.hub_id, self.city, self.state] if p]
        return ", ".join(parts) if parts else "UNKNOWN_LOCATION"

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except Exception as err:
            if isinstance(err, InvalidTrackingEventError):
                raise
            raise InvalidTrackingEventError(str(err)) from err


class ShipmentTrackingEvent(BaseModel):
    """
    Immutable audit record of a single tracking update from a carrier or logistics provider.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    shipment_id: str = Field(..., description="Unique shipment identifier (e.g. SHP-FO-001-01)")
    carrier_id: str = Field(..., description="Carrier or logistics provider identifier (e.g. BLUEDART, SIMULATED)")
    provider_event_id: str = Field(..., description="Carrier-assigned unique event identifier")
    status: ShipmentTrackingStatus = Field(..., description="Shipment tracking status at this checkpoint")
    event_timestamp: str = Field(..., description="ISO 8601 UTC timestamp when physical event occurred")
    received_timestamp: str = Field(..., description="ISO 8601 UTC timestamp when application ingested event")

    location: Optional[TrackingLocation] = Field(default=None, description="Checkpoint location metadata")
    exception_reason: Optional[str] = Field(default=None, description="Reason code if status is DELIVERY_EXCEPTION or RETURN_INITIATED")
    exception_details: Optional[Dict[str, Any]] = Field(default=None, description="Structured diagnostics for exceptions")
    notes: Optional[str] = Field(default=None, description="Carrier status notes or remarks")
    raw_payload: Optional[Dict[str, Any]] = Field(default=None, description="Verbatim provider payload for non-repudiation")

    @field_validator("shipment_id", "carrier_id", "provider_event_id")
    @classmethod
    def validate_ids(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("location", mode="before")
    @classmethod
    def validate_location(cls, v: Any) -> Optional[TrackingLocation]:
        if v is None:
            return None
        if isinstance(v, TrackingLocation):
            return v
        if isinstance(v, dict):
            return TrackingLocation(**v)
        if isinstance(v, str):
            clean = v.strip()
            if not clean:
                return None
            return TrackingLocation(facility_name=clean)
        raise InvalidTrackingEventError(f"Field 'location' must be a TrackingLocation, dict, or string; got {type(v)}.")

    @field_validator("exception_reason", "notes")
    @classmethod
    def validate_optional_strings(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if not clean:
            raise InvalidTrackingEventError(f"Field '{info.field_name}', if provided, cannot be empty or whitespace.")
        return clean

    @field_validator("event_timestamp", "received_timestamp", mode="before")
    @classmethod
    def validate_timestamps(cls, v: Any, info) -> str:
        return validate_and_normalize_iso_timestamp(v, field_name=info.field_name, error_cls=InvalidTrackingEventError)

    @model_validator(mode="after")
    def validate_temporal_and_status_invariants(self) -> "ShipmentTrackingEvent":
        # 1. Received timestamp cannot precede physical event occurrence
        validate_timestamp_sequence(
            self.event_timestamp,
            self.received_timestamp,
            "event_timestamp",
            "received_timestamp",
        )

        # 2. Mandatory exception_reason on exception or return
        if self.status == ShipmentTrackingStatus.DELIVERY_EXCEPTION:
            if not self.exception_reason:
                raise InvalidTrackingEventError(
                    f"Tracking event '{self.provider_event_id}' has status DELIVERY_EXCEPTION but missing 'exception_reason'."
                )

        if self.status == ShipmentTrackingStatus.RETURN_INITIATED:
            if not self.exception_reason and not self.notes:
                raise InvalidTrackingEventError(
                    f"Tracking event '{self.provider_event_id}' has status RETURN_INITIATED but missing 'exception_reason' or 'notes'."
                )

        return self

    @property
    def fingerprint(self) -> str:
        """
        Deterministic SHA-256 fingerprint for strong idempotency and tampering detection.
        """
        loc_dict = self.location.model_dump() if self.location else None
        fp_payload: Dict[str, Any] = {
            "shipment_id": self.shipment_id,
            "carrier_id": self.carrier_id,
            "provider_event_id": self.provider_event_id,
            "status": self.status.value,
            "event_timestamp": self.event_timestamp,
            "location": loc_dict,
            "exception_reason": self.exception_reason,
            "exception_details": self.exception_details,
        }
        return compute_payload_fingerprint(fp_payload)

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except Exception as err:
            if isinstance(err, InvalidTrackingEventError):
                raise
            raise InvalidTrackingEventError(str(err)) from err


class ShipmentTrackingRecord(BaseModel):
    """
    Immutable aggregate tracking record for a single shipment.
    Maintains the current operational tracking status and complete chronological event history.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    shipment_id: str = Field(..., description="Unique shipment identifier")
    carrier_id: str = Field(..., description="Logistics carrier identifier")
    current_status: ShipmentTrackingStatus = Field(
        default=ShipmentTrackingStatus.LABEL_CREATED,
        description="Current operational tracking status"
    )
    events: Tuple[ShipmentTrackingEvent, ...] = Field(
        default_factory=tuple,
        description="Chronological audit history of all tracking events"
    )
    created_at: str = Field(..., description="ISO 8601 UTC timestamp of initial tracking registration")
    updated_at: str = Field(..., description="ISO 8601 UTC timestamp of latest tracking update")
    estimated_delivery_at: Optional[str] = Field(default=None, description="Estimated delivery timestamp")
    actual_delivery_at: Optional[str] = Field(default=None, description="Actual physical delivery timestamp")
    attempt_count: int = Field(default=0, ge=0, description="Count of out-for-delivery delivery attempts")
    scenario: Optional[str] = Field(default=None, description="Deterministic scenario under which tracking was initiated")

    @field_validator("shipment_id", "carrier_id")
    @classmethod
    def validate_ids(cls, v: str, info) -> str:
        return _validate_non_empty_str(v, info.field_name)

    @field_validator("created_at", "updated_at", "estimated_delivery_at", "actual_delivery_at", mode="before")
    @classmethod
    def validate_record_timestamps(cls, v: Any, info) -> Optional[str]:
        if v is None:
            return None
        return validate_and_normalize_iso_timestamp(v, field_name=info.field_name, error_cls=InvalidTrackingEventError)

    @property
    def is_delivered(self) -> bool:
        """True if shipment reached final physical delivery at destination dock."""
        return self.current_status == ShipmentTrackingStatus.DELIVERED

    @property
    def is_returned(self) -> bool:
        """True if shipment was returned to supplier warehouse."""
        return self.current_status == ShipmentTrackingStatus.RETURNED

    @property
    def is_terminal(self) -> bool:
        """True if tracking has reached a terminal state (DELIVERED or RETURNED)."""
        return self.current_status in TERMINAL_TRACKING_STATUSES

    @property
    def has_active_exception(self) -> bool:
        """True if the shipment currently has an unresolved delivery exception."""
        return self.current_status == ShipmentTrackingStatus.DELIVERY_EXCEPTION

    @property
    def latest_event(self) -> Optional[ShipmentTrackingEvent]:
        """Most recent event in chronological history."""
        return self.events[-1] if self.events else None

    @property
    def event_count(self) -> int:
        """Total number of tracking events recorded."""
        return len(self.events)

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except Exception as err:
            if isinstance(err, InvalidTrackingEventError):
                raise
            raise InvalidTrackingEventError(str(err)) from err

    def apply_event(self, event: ShipmentTrackingEvent) -> "ShipmentTrackingRecord":
        """
        Applies an incoming tracking event to the record, enforcing:
        1. Shipment isolation: event.shipment_id must match.
        2. Deduplication & Idempotency: exact provider_event_id + fingerprint is a safe no-op.
        3. Conflict detection: provider_event_id reuse with differing fingerprint raises conflict.
        4. Out-of-order tolerance: late historical events are recorded without regressing status.
        5. Transition rules: valid forward progression for in-order events.
        """
        if event.shipment_id != self.shipment_id:
            raise InvalidTrackingEventError(
                f"Shipment ID mismatch: event belongs to '{event.shipment_id}', but tracking record is for '{self.shipment_id}'."
            )
        if event.carrier_id != self.carrier_id:
            raise InvalidTrackingEventError(
                f"Carrier ID mismatch: event specifies carrier '{event.carrier_id}', but tracking record is registered with carrier '{self.carrier_id}'."
            )

        # 1. Deduplication & Idempotency Check by provider_event_id
        for existing_evt in self.events:
            if existing_evt.provider_event_id == event.provider_event_id:
                if existing_evt.fingerprint == event.fingerprint:
                    # Exact idempotent duplicate: return self without mutating history
                    return self
                raise TrackingIdempotencyConflictError(
                    f"Conflicting provider_event_id '{event.provider_event_id}' for shipment '{self.shipment_id}': "
                    f"existing event has status '{existing_evt.status.value}' at '{existing_evt.event_timestamp}', "
                    f"incoming event has status '{event.status.value}' at '{event.event_timestamp}'."
                )

        # 2. Chronologically latest event in history
        latest_evt = (
            max(self.events, key=lambda e: (e.event_timestamp, e.received_timestamp, e.provider_event_id))
            if self.events else None
        )

        # 3. Terminal State Protection (DELIVERED or RETURNED cannot be transitioned or reopened)
        if self.is_terminal:
            if latest_evt is not None and event.event_timestamp <= latest_evt.event_timestamp:
                # Late-arriving historical event from before/at delivery: accept into audit history without regressing terminal status
                new_events = tuple(sorted(self.events + (event,), key=lambda e: (e.event_timestamp, e.received_timestamp, e.provider_event_id)))
                new_attempt_count = (
                    self.attempt_count + 1
                    if event.status == ShipmentTrackingStatus.OUT_FOR_DELIVERY
                    else self.attempt_count
                )
                new_actual_delivery = next(
                    (e.event_timestamp for e in reversed(new_events) if e.status == ShipmentTrackingStatus.DELIVERED),
                    self.actual_delivery_at,
                )
                new_updated_at = max(self.updated_at, event.received_timestamp)
                return self.model_copy(
                    update={
                        "events": new_events,
                        "updated_at": new_updated_at,
                        "attempt_count": new_attempt_count,
                        "actual_delivery_at": new_actual_delivery,
                    }
                )
            raise InvalidTrackingTransitionError(
                f"Cannot apply tracking event to terminal shipment '{self.shipment_id}' (current status: '{self.current_status.value}')."
            )

        # 4. Out-of-Order Check for non-terminal shipments
        is_out_of_order = False
        if latest_evt is not None:
            if event.event_timestamp < latest_evt.event_timestamp:
                is_out_of_order = True
            elif event.event_timestamp == latest_evt.event_timestamp:
                # Same timestamp: if forward transition is valid, treat as normal forward progression;
                # otherwise, if reverse transition is valid (meaning this event preceded current status), treat as out-of-order historical event.
                if not is_valid_tracking_transition(self.current_status, event.status):
                    if is_valid_tracking_transition(event.status, self.current_status):
                        is_out_of_order = True
                    else:
                        raise InvalidTrackingTransitionError(
                            f"Invalid shipment tracking transition for '{self.shipment_id}': "
                            f"cannot transition from '{self.current_status.value}' to '{event.status.value}'."
                        )

        if is_out_of_order:
            # Out-of-order event: happened in the past relative to latest known progress.
            # Insert into history in chronological sequence without regressing current status.
            new_events = tuple(sorted(self.events + (event,), key=lambda e: (e.event_timestamp, e.received_timestamp, e.provider_event_id)))
            new_attempt_count = (
                self.attempt_count + 1
                if event.status == ShipmentTrackingStatus.OUT_FOR_DELIVERY
                else self.attempt_count
            )
            new_actual_delivery = next(
                (e.event_timestamp for e in reversed(new_events) if e.status == ShipmentTrackingStatus.DELIVERED),
                self.actual_delivery_at,
            )
            new_updated_at = max(self.updated_at, event.received_timestamp)
            return self.model_copy(
                update={
                    "events": new_events,
                    "updated_at": new_updated_at,
                    "attempt_count": new_attempt_count,
                    "actual_delivery_at": new_actual_delivery,
                }
            )

        # 5. In-Order Event: Validate state machine transition
        if not is_valid_tracking_transition(self.current_status, event.status):
            raise InvalidTrackingTransitionError(
                f"Invalid shipment tracking transition for '{self.shipment_id}': "
                f"cannot transition from '{self.current_status.value}' to '{event.status.value}'."
            )

        # 6. Commit in-order update
        new_events = tuple(sorted(self.events + (event,), key=lambda e: (e.event_timestamp, e.received_timestamp, e.provider_event_id)))
        new_attempt_count = (
            self.attempt_count + 1
            if event.status == ShipmentTrackingStatus.OUT_FOR_DELIVERY
            else self.attempt_count
        )
        new_actual_delivery = (
            event.event_timestamp
            if event.status == ShipmentTrackingStatus.DELIVERED
            else self.actual_delivery_at
        )
        new_updated_at = max(self.updated_at, event.received_timestamp)

        return self.model_copy(
            update={
                "current_status": event.status,
                "events": new_events,
                "updated_at": new_updated_at,
                "actual_delivery_at": new_actual_delivery,
                "attempt_count": new_attempt_count,
            }
        )


class ShipmentTrackingResult(BaseModel):
    """
    Validated, immutable result of a tracking query from a provider.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    shipment_id: str = Field(..., description="Unique shipment identifier")
    carrier_id: str = Field(..., description="Logistics carrier identifier")
    status: ShipmentTrackingStatus = Field(..., description="Current operational tracking status")
    record: ShipmentTrackingRecord = Field(..., description="Complete tracking aggregate record")
    queried_at: str = Field(..., description="ISO 8601 UTC query timestamp")
    estimated_delivery_at: Optional[str] = Field(default=None, description="Estimated delivery timestamp")
    latest_event: Optional[ShipmentTrackingEvent] = Field(default=None, description="Most recent tracking event")
    notes: Optional[str] = Field(default=None, description="Operational notes")


# -----------------------------------------------------------------------------
# Tracking Provider Abstraction
# -----------------------------------------------------------------------------

class TrackingProvider(ABC):
    """
    Abstract interface for tracking providers (carrier APIs, aggregators, webhooks).
    Decoupled from state machine and database persistence.
    """

    @abstractmethod
    def register_shipment(
        self,
        shipment_id: str,
        carrier_id: str,
        *,
        scenario: Optional[TrackingScenario] = None,
        base_timestamp: Optional[str] = None,
        estimated_delivery_at: Optional[str] = None,
        **kwargs: Any,
    ) -> ShipmentTrackingRecord:
        """Registers a shipment for tracking."""
        pass

    @abstractmethod
    def get_tracking(
        self,
        shipment_id: str,
        *,
        query_timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> ShipmentTrackingResult:
        """Queries the current tracking state for a shipment."""
        pass

    @abstractmethod
    def submit_event(
        self,
        event: ShipmentTrackingEvent,
        **kwargs: Any,
    ) -> ShipmentTrackingEvent:
        """Ingests an inbound tracking event (e.g. from carrier webhook or polling)."""
        pass


@runtime_checkable
class TrackingProviderProtocol(Protocol):
    """
    Runtime-checkable Protocol for structural subtyping of tracking providers.
    """

    def register_shipment(
        self,
        shipment_id: str,
        carrier_id: str,
        *,
        scenario: Optional[TrackingScenario] = None,
        base_timestamp: Optional[str] = None,
        estimated_delivery_at: Optional[str] = None,
        **kwargs: Any,
    ) -> ShipmentTrackingRecord:
        ...

    def get_tracking(
        self,
        shipment_id: str,
        *,
        query_timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> ShipmentTrackingResult:
        ...

    def submit_event(
        self,
        event: ShipmentTrackingEvent,
        **kwargs: Any,
    ) -> ShipmentTrackingEvent:
        ...


# -----------------------------------------------------------------------------
# Deterministic Simulated Tracking Provider
# -----------------------------------------------------------------------------

class SimulatedTrackingProvider(TrackingProvider):
    """
    Deterministic simulated tracking provider.

    Guarantees:
    - Pure, deterministic outputs: identical inputs and scenarios always yield identical outputs.
    - Zero random numbers, zero real HTTP/network calls, zero external APIs.
    - Scenarios supported:
      1. NORMAL_DELIVERY
      2. DELAYED_DELIVERY
      3. DELIVERY_EXCEPTION_AND_RECOVERY
      4. RETURN_TO_ORIGIN
    - Deduplication of repeated events.
    - Conflicting reuse detection.
    - Strict isolation between separate shipments.
    """

    def __init__(
        self,
        default_scenario: TrackingScenario = TrackingScenario.NORMAL_DELIVERY,
    ) -> None:
        self._default_scenario = default_scenario
        self._shipment_scenarios: Dict[str, TrackingScenario] = {}
        self._carrier_scenarios: Dict[str, TrackingScenario] = {}
        self._shipments: Dict[str, ShipmentTrackingRecord] = {}
        self._scenario_events: Dict[str, List[ShipmentTrackingEvent]] = {}
        self._scenario_step_indices: Dict[str, int] = {}

    def set_default_scenario(self, scenario: TrackingScenario) -> None:
        """Sets the global fallback scenario."""
        self._default_scenario = scenario

    def configure_shipment_scenario(self, shipment_id: str, scenario: TrackingScenario) -> None:
        """Configures a scenario override for a specific shipment ID."""
        clean_id = _validate_non_empty_str(shipment_id, "shipment_id")
        self._shipment_scenarios[clean_id] = scenario

    def configure_carrier_scenario(self, carrier_id: str, scenario: TrackingScenario) -> None:
        """Configures a scenario override for a specific carrier ID."""
        clean_carrier = _validate_non_empty_str(carrier_id, "carrier_id")
        self._carrier_scenarios[clean_carrier] = scenario

    def has_shipment(self, shipment_id: str) -> bool:
        """True if the shipment has been registered or received events."""
        clean_id = shipment_id.strip() if shipment_id else ""
        return clean_id in self._shipments

    def get_all_shipments(self) -> List[str]:
        """Returns all registered shipment identifiers."""
        return sorted(self._shipments.keys())

    def reset(self) -> None:
        """Resets all simulation state and registered shipments."""
        self._shipments.clear()
        self._shipment_scenarios.clear()
        self._carrier_scenarios.clear()
        self._scenario_events.clear()
        self._scenario_step_indices.clear()

    @staticmethod
    def generate_scenario_events(
        shipment_id: str,
        carrier_id: str,
        scenario: TrackingScenario,
        base_timestamp: str,
    ) -> List[ShipmentTrackingEvent]:
        """
        Pure, deterministic generator of tracking events for a given scenario.
        Computes exact timestamps and unique event identifiers without random numbers.
        """
        clean_shipment = _validate_non_empty_str(shipment_id, "shipment_id")
        clean_carrier = _validate_non_empty_str(carrier_id, "carrier_id")
        clean_base = validate_and_normalize_iso_timestamp(base_timestamp, "base_timestamp")
        dt_base = dt_datetime.fromisoformat(clean_base.replace("Z", "+00:00"))

        events: List[ShipmentTrackingEvent] = []

        def _make_event(
            seq: int,
            status: ShipmentTrackingStatus,
            offset_hours: float,
            hub_id: str,
            facility_name: str,
            city: str = "Bangalore",
            exception_reason: Optional[str] = None,
            exception_details: Optional[Dict[str, Any]] = None,
            notes: Optional[str] = None,
        ) -> ShipmentTrackingEvent:
            evt_dt = dt_base + timedelta(hours=offset_hours)
            rec_dt = evt_dt + timedelta(minutes=5)
            evt_ts = evt_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            rec_ts = rec_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            event_id = f"EVT-{clean_carrier}-{clean_shipment}-{seq:03d}"

            return ShipmentTrackingEvent(
                shipment_id=clean_shipment,
                carrier_id=clean_carrier,
                provider_event_id=event_id,
                status=status,
                event_timestamp=evt_ts,
                received_timestamp=rec_ts,
                location=TrackingLocation(hub_id=hub_id, facility_name=facility_name, city=city),
                exception_reason=exception_reason,
                exception_details=exception_details,
                notes=notes,
            )

        if scenario == TrackingScenario.NORMAL_DELIVERY:
            events = [
                _make_event(1, ShipmentTrackingStatus.LABEL_CREATED, 0.0, "HUB-VND-01", "Supplier Dispatch Bay", notes="Shipping label created"),
                _make_event(2, ShipmentTrackingStatus.PICKED_UP, 2.0, "HUB-VND-01", "Supplier Dispatch Bay", notes="Shipment picked up by carrier vehicle"),
                _make_event(3, ShipmentTrackingStatus.IN_TRANSIT, 4.0, "CORR-NH44", "NH44 Line-Haul Corridor", notes="In transit to destination city hub"),
                _make_event(4, ShipmentTrackingStatus.AT_HUB, 8.0, "HUB-BLR-NORTH", "Bangalore North Sorting Facility", notes="Consignment sorted at regional hub"),
                _make_event(5, ShipmentTrackingStatus.OUT_FOR_DELIVERY, 12.0, "HUB-BLR-NORTH", "Bangalore North Sorting Facility", notes="Out for delivery to dark-store dock"),
                _make_event(6, ShipmentTrackingStatus.DELIVERED, 14.0, "DS-BLR-DOCK", "Dark Store Receiving Bay", notes="Delivered at receiving bay"),
            ]

        elif scenario == TrackingScenario.DELAYED_DELIVERY:
            events = [
                _make_event(1, ShipmentTrackingStatus.LABEL_CREATED, 0.0, "HUB-VND-01", "Supplier Dispatch Bay"),
                _make_event(2, ShipmentTrackingStatus.PICKED_UP, 2.0, "HUB-VND-01", "Supplier Dispatch Bay"),
                _make_event(3, ShipmentTrackingStatus.IN_TRANSIT, 4.0, "CORR-NH44", "NH44 Line-Haul Corridor"),
                _make_event(4, ShipmentTrackingStatus.AT_HUB, 8.0, "HUB-HYD-01", "Hyderabad Central Hub", notes="Intermediate sorting hub transit"),
                _make_event(5, ShipmentTrackingStatus.IN_TRANSIT, 16.0, "CORR-NH44-S", "South Line-Haul Corridor", notes="Secondary transfer line-haul"),
                _make_event(6, ShipmentTrackingStatus.AT_HUB, 24.0, "HUB-BLR-NORTH", "Bangalore North Sorting Facility"),
                _make_event(7, ShipmentTrackingStatus.OUT_FOR_DELIVERY, 30.0, "HUB-BLR-NORTH", "Bangalore North Sorting Facility"),
                _make_event(8, ShipmentTrackingStatus.DELIVERED, 32.0, "DS-BLR-DOCK", "Dark Store Receiving Bay", notes="Delayed delivery completed"),
            ]

        elif scenario == TrackingScenario.DELIVERY_EXCEPTION_AND_RECOVERY:
            events = [
                _make_event(1, ShipmentTrackingStatus.LABEL_CREATED, 0.0, "HUB-VND-01", "Supplier Dispatch Bay"),
                _make_event(2, ShipmentTrackingStatus.PICKED_UP, 2.0, "HUB-VND-01", "Supplier Dispatch Bay"),
                _make_event(3, ShipmentTrackingStatus.IN_TRANSIT, 4.0, "CORR-NH44", "NH44 Line-Haul Corridor"),
                _make_event(4, ShipmentTrackingStatus.AT_HUB, 8.0, "HUB-BLR-NORTH", "Bangalore North Sorting Facility"),
                _make_event(5, ShipmentTrackingStatus.OUT_FOR_DELIVERY, 12.0, "HUB-BLR-NORTH", "Bangalore North Sorting Facility", notes="First delivery attempt"),
                _make_event(
                    6, ShipmentTrackingStatus.DELIVERY_EXCEPTION, 14.0, "DS-BLR-DOCK", "Dark Store Receiving Bay",
                    exception_reason="DOCK_BAY_CONGESTED",
                    exception_details={"attempt": 1, "bay_status": "CONGESTED", "retry_scheduled": True},
                    notes="Receiving dock congested; delivery vehicle held or rescheduled",
                ),
                _make_event(7, ShipmentTrackingStatus.AT_HUB, 16.0, "HUB-BLR-NORTH", "Bangalore North Sorting Facility", notes="Returned to hub pending morning re-dispatch"),
                _make_event(8, ShipmentTrackingStatus.OUT_FOR_DELIVERY, 24.0, "HUB-BLR-NORTH", "Bangalore North Sorting Facility", notes="Second delivery attempt"),
                _make_event(9, ShipmentTrackingStatus.DELIVERED, 26.0, "DS-BLR-DOCK", "Dark Store Receiving Bay", notes="Delivered successfully on attempt 2"),
            ]

        elif scenario == TrackingScenario.RETURN_TO_ORIGIN:
            events = [
                _make_event(1, ShipmentTrackingStatus.LABEL_CREATED, 0.0, "HUB-VND-01", "Supplier Dispatch Bay"),
                _make_event(2, ShipmentTrackingStatus.PICKED_UP, 2.0, "HUB-VND-01", "Supplier Dispatch Bay"),
                _make_event(3, ShipmentTrackingStatus.IN_TRANSIT, 4.0, "CORR-NH44", "NH44 Line-Haul Corridor"),
                _make_event(4, ShipmentTrackingStatus.AT_HUB, 8.0, "HUB-BLR-NORTH", "Bangalore North Sorting Facility"),
                _make_event(5, ShipmentTrackingStatus.OUT_FOR_DELIVERY, 12.0, "HUB-BLR-NORTH", "Bangalore North Sorting Facility"),
                _make_event(
                    6, ShipmentTrackingStatus.DELIVERY_EXCEPTION, 14.0, "DS-BLR-DOCK", "Dark Store Receiving Bay",
                    exception_reason="STORE_REFUSED_DELIVERY",
                    exception_details={"refusal_reason": "UNSCHEDULED_PALLET_SURCHARGE"},
                    notes="Store refused delivery consignment",
                ),
                _make_event(
                    7, ShipmentTrackingStatus.RETURN_INITIATED, 16.0, "HUB-BLR-NORTH", "Bangalore North Sorting Facility",
                    exception_reason="RTO_REQUESTED",
                    notes="Consignment authorized for return to supplier",
                ),
                _make_event(8, ShipmentTrackingStatus.IN_TRANSIT, 20.0, "CORR-NH44-RET", "Return Corridor", notes="In transit back to vendor"),
                _make_event(9, ShipmentTrackingStatus.AT_HUB, 24.0, "HUB-VND-01", "Supplier Dispatch Bay", notes="Arrived at vendor return hub"),
                _make_event(10, ShipmentTrackingStatus.RETURNED, 28.0, "HUB-VND-01", "Supplier Dispatch Bay", notes="Returned to origin supplier warehouse"),
            ]

        return events

    def register_shipment(
        self,
        shipment_id: str,
        carrier_id: str = "SIM_CARRIER",
        *,
        scenario: Optional[TrackingScenario] = None,
        base_timestamp: Optional[str] = None,
        estimated_delivery_at: Optional[str] = None,
        **kwargs: Any,
    ) -> ShipmentTrackingRecord:
        """
        Registers a shipment and initializes its tracking record and scenario event pipeline.
        """
        clean_shipment = _validate_non_empty_str(shipment_id, "shipment_id")
        clean_carrier = _validate_non_empty_str(carrier_id, "carrier_id")
        clean_base = validate_and_normalize_iso_timestamp(
            base_timestamp or "2024-11-15T08:00:00Z",
            "base_timestamp",
        )

        effective_scenario = (
            scenario
            or self._shipment_scenarios.get(clean_shipment)
            or self._carrier_scenarios.get(clean_carrier)
            or self._default_scenario
        )

        if clean_shipment in self._shipments:
            existing = self._shipments[clean_shipment]
            if existing.carrier_id != clean_carrier:
                raise TrackingIdempotencyConflictError(
                    f"Shipment '{clean_shipment}' already registered with carrier '{existing.carrier_id}', "
                    f"cannot re-register with '{clean_carrier}'."
                )
            return existing

        # Generate scenario events
        events_pipeline = self.generate_scenario_events(
            clean_shipment,
            clean_carrier,
            effective_scenario,
            clean_base,
        )

        # First event is LABEL_CREATED
        first_event = events_pipeline[0] if events_pipeline else None
        est_del = estimated_delivery_at
        if not est_del and events_pipeline:
            # Estimate delivery as timestamp of the last delivery event
            est_del = events_pipeline[-1].event_timestamp

        record = ShipmentTrackingRecord(
            shipment_id=clean_shipment,
            carrier_id=clean_carrier,
            current_status=first_event.status if first_event else ShipmentTrackingStatus.LABEL_CREATED,
            events=(first_event,) if first_event else (),
            created_at=clean_base,
            updated_at=first_event.received_timestamp if first_event else clean_base,
            estimated_delivery_at=est_del,
            actual_delivery_at=None,
            attempt_count=0,
            scenario=effective_scenario.value if effective_scenario else None,
        )

        self._shipments[clean_shipment] = record
        self._scenario_events[clean_shipment] = events_pipeline
        self._scenario_step_indices[clean_shipment] = 1 if first_event else 0

        return record

    def get_tracking(
        self,
        shipment_id: str,
        *,
        query_timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> ShipmentTrackingResult:
        """
        Retrieves current tracking status and event history for a registered shipment.
        """
        clean_shipment = _validate_non_empty_str(shipment_id, "shipment_id")
        if clean_shipment not in self._shipments:
            raise ShipmentNotFoundError(f"Shipment '{clean_shipment}' is not registered with tracking provider.")

        record = self._shipments[clean_shipment]
        q_ts = validate_and_normalize_iso_timestamp(
            query_timestamp or record.updated_at,
            "query_timestamp",
        )

        return ShipmentTrackingResult(
            shipment_id=record.shipment_id,
            carrier_id=record.carrier_id,
            status=record.current_status,
            record=record,
            queried_at=q_ts,
            estimated_delivery_at=record.estimated_delivery_at,
            latest_event=record.latest_event,
            notes=f"Simulated tracking status: {record.current_status.value}",
        )

    def submit_event(
        self,
        event: ShipmentTrackingEvent,
        **kwargs: Any,
    ) -> ShipmentTrackingEvent:
        """
        Ingests an inbound tracking event (e.g. from a carrier webhook or external API).
        Applies deduplication, conflict checking, and out-of-order handling.
        """
        if not isinstance(event, ShipmentTrackingEvent):
            raise InvalidTrackingEventError(f"Expected ShipmentTrackingEvent, got {type(event)}.")

        shipment_id = event.shipment_id

        # Auto-register if shipment record not present
        if shipment_id not in self._shipments:
            initial_record = ShipmentTrackingRecord(
                shipment_id=shipment_id,
                carrier_id=event.carrier_id,
                current_status=event.status,
                events=(),
                created_at=event.event_timestamp,
                updated_at=event.received_timestamp,
            )
            self._shipments[shipment_id] = initial_record

        current_record = self._shipments[shipment_id]
        updated_record = current_record.apply_event(event)
        self._shipments[shipment_id] = updated_record

        return event

    def advance_shipment(
        self,
        shipment_id: str,
        steps: int = 1,
    ) -> Optional[ShipmentTrackingEvent]:
        """
        Steps forward through the configured scenario pipeline for a registered shipment.
        Applies the next event(s) and returns the latest applied event.
        Returns None if the scenario pipeline is already exhausted.
        """
        clean_shipment = _validate_non_empty_str(shipment_id, "shipment_id")
        if clean_shipment not in self._shipments:
            raise ShipmentNotFoundError(f"Shipment '{clean_shipment}' is not registered with tracking provider.")

        pipeline = self._scenario_events.get(clean_shipment, [])
        curr_idx = self._scenario_step_indices.get(clean_shipment, 0)

        last_applied: Optional[ShipmentTrackingEvent] = None
        for _ in range(max(1, steps)):
            if curr_idx >= len(pipeline):
                break
            next_event = pipeline[curr_idx]
            self.submit_event(next_event)
            curr_idx += 1
            last_applied = next_event

        self._scenario_step_indices[clean_shipment] = curr_idx
        return last_applied

    def advance_to_status(
        self,
        shipment_id: str,
        target_status: ShipmentTrackingStatus,
    ) -> List[ShipmentTrackingEvent]:
        """
        Advances the shipment along its scenario pipeline until target_status is reached.
        Returns all events applied during the progression.
        """
        clean_shipment = _validate_non_empty_str(shipment_id, "shipment_id")
        if clean_shipment not in self._shipments:
            raise ShipmentNotFoundError(f"Shipment '{clean_shipment}' is not registered with tracking provider.")

        pipeline = self._scenario_events.get(clean_shipment, [])
        curr_idx = self._scenario_step_indices.get(clean_shipment, 0)

        # Check if target_status is present in remaining pipeline
        target_found = any(evt.status == target_status for evt in pipeline[curr_idx:])
        if not target_found:
            # If current status is already target_status, return empty list
            if self._shipments[clean_shipment].current_status == target_status:
                return []
            raise InvalidTrackingTransitionError(
                f"Target status '{target_status.value}' does not exist in remaining pipeline for shipment '{clean_shipment}'."
            )

        applied: List[ShipmentTrackingEvent] = []
        while curr_idx < len(pipeline):
            next_event = pipeline[curr_idx]
            self.submit_event(next_event)
            applied.append(next_event)
            curr_idx += 1
            if next_event.status == target_status:
                break

        self._scenario_step_indices[clean_shipment] = curr_idx
        return applied


# -----------------------------------------------------------------------------
# Critical Business Boundary Invariant Helpers
# -----------------------------------------------------------------------------

def assert_tracking_does_not_mutate_fulfilment_order(
    fo_before: FulfilmentOrder,
    fo_after: FulfilmentOrder,
    po_before: Optional[Any] = None,
    po_after: Optional[Any] = None,
) -> None:
    """
    Enforces the critical Phase 5.5.1 architectural boundary:
    A tracking event (even DELIVERED) must NEVER independently mutate a FulfilmentOrder's
    goods receipt notes, accepted quantities, delivered quantities, inventory batches,
    inbound pipeline quantities, or the approved PurchaseOrder.
    """
    if fo_before.po_id != fo_after.po_id:
        raise InvalidTrackingEventError("Boundary Violation: Tracking update altered po_id reference.")

    if po_before is not None and po_after is not None:
        if po_before != po_after:
            raise InvalidTrackingEventError("Boundary Violation: Tracking update mutated approved PurchaseOrder.")

    if fo_before.grn_ids != fo_after.grn_ids:
        raise InvalidTrackingEventError("Boundary Violation: Tracking update created or altered GoodsReceiptNote IDs.")

    if fo_before.total_received_accepted_qty != fo_after.total_received_accepted_qty:
        raise InvalidTrackingEventError("Boundary Violation: Tracking update altered total_received_accepted_qty.")

    if fo_before.total_confirmed_pipeline_qty != fo_after.total_confirmed_pipeline_qty:
        raise InvalidTrackingEventError("Boundary Violation: Tracking update altered total_confirmed_pipeline_qty.")

    if fo_before.total_remaining_unfulfilled_qty != fo_after.total_remaining_unfulfilled_qty:
        raise InvalidTrackingEventError("Boundary Violation: Tracking update altered total_remaining_unfulfilled_qty.")

    for l_b, l_a in zip(fo_before.lines, fo_after.lines):
        if l_b.delivered_quantity != l_a.delivered_quantity:
            raise InvalidTrackingEventError("Boundary Violation: Tracking update altered delivered_quantity.")
        if l_b.accepted_quantity != l_a.accepted_quantity:
            raise InvalidTrackingEventError("Boundary Violation: Tracking update altered accepted_quantity.")
