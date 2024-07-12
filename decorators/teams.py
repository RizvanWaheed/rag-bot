from fastapi import HTTPException, Request
from functools import wraps

def x_super_team(fn):
    @wraps(fn)
    async def teamdecorator(request: Request):
        x_super_team = request.headers.get('x-super-team')

        if x_super_team and x_super_team != 0:
            return await fn(request)
        else:
            return HTTPException(status_code = 401, detail = "Invalid or missing super team!")
    return teamdecorator