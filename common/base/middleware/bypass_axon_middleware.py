import bittensor as bt
from bittensor.core.axon import Axon, AxonMiddleware
from starlette.middleware import Middleware


class BypassAxonMiddleware(AxonMiddleware):
    def __init__(self, app, axon: Axon, exclude_paths: list[str] = []):
        super().__init__(app, axon)
        self.exclude_paths = exclude_paths

    async def dispatch(self, request, call_next):
        if request.url.path in self.exclude_paths:
            bt.logging.debug(
                f"BypassAxonMiddleware: Bypassing AxonMiddleware logic for path {request.url.path}"
            )
            return await call_next(request)
        bt.logging.debug(
            f"BypassAxonMiddleware: Processing request for path {request.url.path}"
        )
        return await super().dispatch(request, call_next)


def replace_axon_middleware(axon: Axon, exclude_paths: list[str] = []):
    """
    Replace the AxonMiddleware with BypassAxonMiddleware in the Axon instance.
    This allows for specific paths to bypass the Axon middleware logic.
    """
    # Remove existing middleware
    bt.logging.debug(f"Remove AxonMiddleware, Before: {axon.app.user_middleware}")
    new_middlewares: list[Middleware] = []
    for middleware in axon.app.user_middleware:
        if not middleware.cls.__name__ == "AxonMiddleware":
            new_middlewares.append(middleware)
    axon.app.user_middleware = new_middlewares
    bt.logging.debug(f"Remove AxonMiddleware, After: {axon.app.user_middleware}")

    # Add the new BypassAxonMiddleware
    axon.app.add_middleware(
        BypassAxonMiddleware,
        axon=axon,
        exclude_paths=exclude_paths,
    )
    bt.logging.debug(
        f"Added BypassAxonMiddleware, Current middlewares: {axon.app.user_middleware}"
    )
