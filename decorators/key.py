from fastapi import HTTPException, Request
from functools import wraps
from decouple import config

key = config("X_APP_KEY")

def x_app_key(fn):
    @wraps(fn)
    async def apidecorator(request: Request):
        x_app_key = request.headers.get('x-app-key')

        if x_app_key == key:
            return await fn(request)
        else:
            raise HTTPException(status_code = 401, detail = "Invalid or missing app key!")
    return apidecorator