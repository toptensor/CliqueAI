from typing import Generic, TypeVar

from pydantic import BaseModel

PayloadType = TypeVar("PayloadType", bound=BaseModel)


class SignedRequest(BaseModel, Generic[PayloadType]):
    payload: PayloadType
    signature: str
