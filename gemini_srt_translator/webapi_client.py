"""
Synchronous wrapper for gemini_webapi.GeminiClient.

Provides a unified interface so the translator can work with the
Gemini Web API (browser-cookie-based, no API key required) as an
alternative backend.

The gemini_webapi package is fully async; this module bridges the
gap by running an asyncio event loop in a background daemon thread
and exposing synchronous methods.
"""

import asyncio
import io
import queue
import threading
from typing import Generator, Optional

try:
    from gemini_webapi import GeminiClient
    from gemini_webapi.types import ModelOutput

    WEBAPI_AVAILABLE = True
except ImportError:
    WEBAPI_AVAILABLE = False


class WebAPIClientWrapper:
    """
    Synchronous wrapper around the async ``GeminiClient``.

    Parameters
    ----------
    secure_1psid : str
        ``__Secure-1PSID`` cookie value from gemini.google.com.
    secure_1psidts : str, optional
        ``__Secure-1PSIDTS`` cookie value (not all accounts need this).
    proxy : str, optional
        Proxy URL (e.g. ``http://host:port``).
    """

    def __init__(
        self,
        secure_1psid: Optional[str] = None,
        secure_1psidts: Optional[str] = None,
        proxy: Optional[str] = None,
        browser: bool = False,
    ):
        if not WEBAPI_AVAILABLE:
            raise ImportError(
                "gemini-webapi package is not installed. "
                "Install it with: pip install gemini-webapi"
            )

        if browser:
            try:
                import browser_cookie3
            except ImportError:
                raise ImportError(
                    "browser_cookie3 package is required for browser cookie extraction. "
                    "Install it with: pip install browser_cookie3 or pip install gemini-webapi[browser]"
                )

        # Create a dedicated event loop running in a daemon thread.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._thread.start()

        # Initialise the async client synchronously.
        if browser and not secure_1psid:
            self._client = GeminiClient(proxy=proxy)
            self._run_async(self._client.init(timeout=30, auto_close=False, auto_refresh=True, verbose=False))
        else:
            self._client = GeminiClient(
                secure_1psid, secure_1psidts, proxy=proxy
            )
            self._run_async(self._client.init(verbose=False))

        # Chat session (optional, created via start_chat)
        self._chat = None

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _run_async(self, coro):
        """Submit a coroutine to the background loop and block until done."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # --------------------------------------------------------------------- #
    # Content generation
    # --------------------------------------------------------------------- #

    def generate_content(
        self,
        prompt: str,
        model: str = "unspecified",
        files: list = None,
    ):
        """
        Synchronous single-turn content generation.

        Returns
        -------
        ModelOutput
            The full model output; use ``.text`` for the response text
            and ``.thoughts`` for the thinking process.
        """
        return self._run_async(
            self._client.generate_content(
                prompt, model=model, files=files, temporary=True
            )
        )

    def generate_content_stream(
        self,
        prompt: str,
        model: str = "unspecified",
        files: list = None,
    ) -> Generator:
        """
        Synchronous streaming content generation.

        Yields ``ModelOutput`` chunks with ``.text_delta`` containing
        only the newly received characters.
        """
        q: queue.Queue = queue.Queue()

        async def _stream():
            try:
                async for chunk in self._client.generate_content_stream(
                    prompt, model=model, files=files, temporary=True
                ):
                    q.put(chunk)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(None)  # sentinel

        future = asyncio.run_coroutine_threadsafe(_stream(), self._loop)

        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        # Propagate any exception from the task itself.
        future.result()

    # --------------------------------------------------------------------- #
    # Chat sessions
    # --------------------------------------------------------------------- #

    def start_chat(self, model: str = "unspecified"):
        """
        Start a multi-turn chat session.

        The chat object is stored internally; use :meth:`send_message`
        and :meth:`send_message_stream` to interact with it.
        """
        self._chat = self._client.start_chat(model=model)
        return self._chat

    def send_message(self, message: str, files: list = None):
        """Send a message within the current chat session (blocking)."""
        if self._chat is None:
            raise RuntimeError("No active chat session. Call start_chat() first.")
        return self._run_async(
            self._chat.send_message(message, files=files)
        )

    def send_message_stream(
        self, message: str, files: list = None
    ) -> Generator:
        """
        Send a message within the current chat session (streaming).

        Yields ``ModelOutput`` chunks.
        """
        if self._chat is None:
            raise RuntimeError("No active chat session. Call start_chat() first.")

        q: queue.Queue = queue.Queue()

        async def _stream():
            try:
                async for chunk in self._chat.send_message_stream(
                    message, files=files
                ):
                    q.put(chunk)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(None)

        future = asyncio.run_coroutine_threadsafe(_stream(), self._loop)

        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        future.result()

    # --------------------------------------------------------------------- #
    # Model listing
    # --------------------------------------------------------------------- #

    def list_models(self) -> list:
        """
        List available models for the current account.

        Returns
        -------
        list
            List of ``AvailableModel`` objects (or ``None`` if unavailable).
        """
        return self._client.list_models()

    # --------------------------------------------------------------------- #
    # Cleanup
    # --------------------------------------------------------------------- #

    def close(self):
        """Close the underlying async client and stop the event loop."""
        try:
            self._run_async(self._client.close())
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()
