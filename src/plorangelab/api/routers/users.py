from fastapi import APIRouter, HTTPException, Response, status

from plorangelab.services.users import UsernameAlreadyExistsError, register_user

from ..dependencies import DbSession
from ..schemas import RegisterUserRequest

router = APIRouter(
    prefix="/api/users",
    tags=["users"],
)


@router.post("", status_code=status.HTTP_201_CREATED, response_class=Response)
async def create_user(db_session: DbSession, request: RegisterUserRequest) -> Response:
    try:
        await register_user(
            db_session, request.username, request.password.get_secret_value()
        )
    except UsernameAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        ) from exc

    return Response(status_code=status.HTTP_201_CREATED)
