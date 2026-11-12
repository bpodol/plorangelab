"""REST endpoints for managing player filters and nested filter groups.

Supports creating, activating, deactivating, deleting, and retrieving
filters and groups for a player's active street.

Filter evaluation and state management are delegated to RangeManager.
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import (
    PlayerId,
    SessionDep,
    editable_session,
    locked_session,
)
from ..schemas import (
    FilterCoverageResponse,
    FilterCreateRequest,
    FilterGroupInfoResponse,
    FilterInfoResponse,
    GroupCreateRequest,
    GroupResponse,
    SetActiveRequest,
)

# Nested group paths are passed as repeated query parameters:
# ?path=outer&path=inner identifies the parent group hierarchy.
FilterPath = Annotated[tuple[str, ...], Query()]

router = APIRouter(
    prefix="/api/sessions/{session_id}/players/{player_id}",
    tags=["filters"],
)


@router.post("/filters", status_code=201, response_model=FilterCoverageResponse)
def create_filter(
    player_id: PlayerId,
    request: FilterCreateRequest,
    session: SessionDep,
):
    """Create a filter in the player's active street.

    Evaluates the supplied expression against the street's base
    range and saves the resulting selection.

    The optional name defaults to the filter expression. The path
    identifies the parent group, or the root when omitted.

    Returns:
        HTTP 201 with the new filter's weighted membership coverage
        relative to the active street's base range.

    Raises:
        HTTPException: 404 if the session is missing or expired.
        HTTPException: 409 if an equity calculation is unfinished.
        HTTPException: 422 if the expression, name, or path is invalid,
            or the target location already contains the name.
    """

    with editable_session(session):
        try:
            coverage = session.manager.add_filter(
                expression=request.expression,
                player_index=player_id,
                name=request.name,
                path=request.path,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return FilterCoverageResponse(coverage=coverage)


@router.post("/groups", status_code=201, response_model=GroupResponse)
def create_group(
    player_id: PlayerId,
    request: GroupCreateRequest,
    session: SessionDep,
):
    """Create an empty filter group at the specified path.

    Groups may contain individual filters and other groups.
    New groups start inactive, even when their parent is active.

    Returns:
        HTTP 201 with the new group's name and parent path.

    Raises:
        HTTPException: 404 if the session is missing or expired.
        HTTPException: 409 if an equity calculation is unfinished.
        HTTPException: 422 if the group name or path is invalid,
            or the name already exists at the target location.
    """

    with editable_session(session):
        try:
            session.manager.add_group(
                name=request.name,
                player_index=player_id,
                path=request.path,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return GroupResponse(name=request.name, path=request.path)


@router.patch("/groups/{name}", response_model=FilterCoverageResponse)
@router.patch("/filters/{name}", response_model=FilterCoverageResponse)
def set_filter_active(
    player_id: PlayerId,
    name: str,
    request: SetActiveRequest,
    session: SessionDep,
    path: FilterPath = (),
):
    """Activate or deactivate a saved filter or group.

    The optional query path identifies the target's parent groups.
    Activating a filter also activates its ancestors; activating a
    group activates all descendants.

    Deactivating a group recursively deactivates its descendants.

    Returns:
        HTTP 200 with the player's resulting effective weighted
        coverage for the active street.

    Raises:
        HTTPException: 404 if the session is missing or expired.
        HTTPException: 409 if an equity calculation is unfinished.
        HTTPException: 422 if the target does not exist or cannot
            be activated because it matches no hands.
    """

    with editable_session(session):
        try:
            coverage = session.manager.set_filter_active(
                name=name,
                player_index=player_id,
                active=request.active,
                path=path,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return FilterCoverageResponse(coverage=coverage)


@router.post("/filters/clear", response_model=FilterCoverageResponse)
def clear_filters(player_id: PlayerId, session: SessionDep):
    """Deactivate all filters and groups in the active street.

    Preserves saved filter expressions, groups, and their hierarchy.
    The player's effective range returns to the street's base range.

    Returns:
        HTTP 200 with effective weighted coverage of 1.0.

    Raises:
        HTTPException: 404 if the session is missing or expired.
        HTTPException: 409 if an equity calculation is unfinished.
        HTTPException: 422 if the player slot is empty.
    """

    with editable_session(session):
        try:
            coverage = session.manager.clear_filters(player_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return FilterCoverageResponse(coverage=coverage)


# Both URLs delegate to the same tree operation; the manager
# resolves the node by its name and parent path.
@router.delete("/filters/{name}", response_model=FilterCoverageResponse)
@router.delete("/groups/{name}", response_model=FilterCoverageResponse)
def delete_filter_or_group(
    player_id: PlayerId,
    name: str,
    session: SessionDep,
    path: FilterPath = (),
):
    """Delete a filter or group from the player's active street.

    Deleting a group also removes all its descendants. The optional
    query path identifies the target's parent group hierarchy.

    The resulting active range is recalculated after deletion.

    Returns:
        HTTP 200 with the updated effective weighted coverage.

    Raises:
        HTTPException: 404 if the session is missing or expired.
        HTTPException: 409 if an equity calculation is unfinished.
        HTTPException: 422 if the target or its parent path does
            not exist.
    """

    with editable_session(session):
        try:
            coverage = session.manager.delete_filter(name, player_id, path)
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

        return FilterCoverageResponse(coverage=coverage)


@router.get(
    "/filters", response_model=dict[str, FilterInfoResponse | FilterGroupInfoResponse]
)
def get_filters(player_id: PlayerId, session: SessionDep):
    """Retrieve the complete filter hierarchy for the active street.

    Returns a dictionary keyed by root-level filter and group names.
    Groups contain recursively serialized child filters and groups.

    Each node includes its type, name, coverage, and activation state.
    Individual filters also include their expressions.

    Returns:
        HTTP 200 with the serialized filter hierarchy.

    Raises:
        HTTPException: 404 if the session or player slot does not exist.
    """

    with locked_session(session):
        try:
            return session.manager.get_filters(player_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/filters/{name}", response_model=FilterInfoResponse | FilterGroupInfoResponse
)
@router.get(
    "/groups/{name}", response_model=FilterInfoResponse | FilterGroupInfoResponse
)
def get_filter(
    player_id: PlayerId,
    name: str,
    session: SessionDep,
    path: FilterPath = (),
):
    """Retrieve metadata for one saved filter or group.

    The optional query path identifies the target's parent groups.
    Group responses include their recursively nested descendants.

    Returns:
        HTTP 200 with filter or group metadata.

    Raises:
        HTTPException: 404 if the session, player slot, or target
            filter or group does not exist.
    """

    with locked_session(session):
        try:
            return session.manager.get_filter(
                name=name,
                player_index=player_id,
                path=path,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
