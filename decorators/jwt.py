import jwt
from fastapi import HTTPException, Request
from functools import wraps
from decouple import config

salt = config("TOKEN_SALT")

def jwt_token(fn):
    @wraps(fn)
    async def decorated(request: Request):

        headers = request.headers

        if "Authorization" in headers:
            token_from_request = headers.get('Authorization').split(" ")[1]

        if not token_from_request:
            HTTPException(status_code = 401, detail="Missing authorization token!")
      
        try:
            options = {
            'verify_exp': False, 
            'verify_aud': False 
            }

            payload = jwt.decode(token_from_request, salt, 'HS256', options)
            request.state.current_user = payload['UserId']

        except:
            HTTPException(status_code = 401, detail="invalid authorization token!")

        return await fn(request)

    return decorated