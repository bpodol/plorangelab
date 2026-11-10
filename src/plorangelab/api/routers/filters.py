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


@router.post("/filters/clear",  response_model=FilterCoverageResponse)
def clear_filters(player_id: PlayerId, session: SessionDep):
    with editable_session(session):
        try:
            coverage = session.manager.clear_filters(player_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return FilterCoverageResponse(coverage=coverage)


@router.delete("/filters/{name}", response_model=FilterCoverageResponse)
@router.delete("/groups/{name}", response_model=FilterCoverageResponse)
def delete_filter_or_group(
    player_id: PlayerId,
    name: str,
    session: SessionDep,
    path: FilterPath = (),
):
    with editable_session(session):
        try:
            coverage = session.manager.delete_filter(name, player_id, path)
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

        return FilterCoverageResponse(coverage=coverage)


@router.get("/filters", response_model=dict[str, FilterInfoResponse | FilterGroupInfoResponse])
def get_filters(player_id: PlayerId, session: SessionDep):
    with locked_session(session):
        try:
            return session.manager.get_filters(player_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/filters/{name}", response_model=FilterInfoResponse | FilterGroupInfoResponse)
@router.get("/groups/{name}", response_model=FilterInfoResponse | FilterGroupInfoResponse)
def get_filter(
    player_id: PlayerId,
    name: str,
    session: SessionDep,
    path: FilterPath = (),
):
    with locked_session(session):
        try:
            return session.manager.get_filter(
                name=name,
                player_index=player_id,
                path=path,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc