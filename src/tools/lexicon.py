import pathlib
import httpx
import pydantic
import typing as typ
from aiofilecache import FileCache
from aiocache.serializers import PickleSerializer
from throughster.base import RetryingConstructor, get_default_retry
from throughster.core import decorators


UmlsLanguages = typ.Literal[
    "ENG",
    "FRE",
    "SWE",
    "CZE",
    "FIN",
    "GER",
    "ITA",
    "JPN",
    "POL",
    "POR",
    "RUS",
    "SPA",
    "SCR",
    "NOR",
    "DUT",
    "LAV",
    "ARA",
    "EST",
    "GRE",
    "HUN",
    "KOR",
    "BAQ",
    "DAN",
    "HEB",
    "CHI",
    "TUR",
    "UKR",
]


class CUI(pydantic.BaseModel):
    """{
    "ui": "C0009044",
    "rootSource": "SNOMEDCT_US",
    "uri": "https://uts-ws.nlm.nih.gov/rest/content/2015AA/CUI/C0009044",
    "name": "Closed fracture carpal bone"
    }
    """

    ui: str = pydantic.Field(..., description="The CUI")
    rootSource: str = pydantic.Field(..., description="The root source")
    uri: str = pydantic.Field(..., description="The URI")
    name: str = pydantic.Field(..., description="The name")


def is_last_page(results: list[dict[str, str]], page_size: int) -> bool:
    if len(results) == 0:
        return True
    if len(results) < page_size:
        return True
    if results[0].get("ui") == "NONE" and results[0].get("name") == "NO RESULTS":
        return True
    return False


def interpret_source_type(result):
    if isinstance(result, dict):
        return result.get("results", [])
    return result


def k_greater_than_results(k, results):
    if not k:
        return True
    return k > len(results)


def _get_data(client: httpx.Client, endpoint: str, request: dict[str, str], k=None):
    results = []
    page = 1

    while True:
        if not k_greater_than_results(k, results):
            break
        request.update(pageNumber=page)
        response = client.get(endpoint, params=request)
        response.raise_for_status()
        items = response.json()

        if "result" not in items:
            break
        res = interpret_source_type(items["result"])
        if is_last_page(res, items["pageSize"]):
            break

        results += res
        page += 1

    if not k_greater_than_results(k, results):
        results = results[:k]
    return results


class UMLS:
    def __init__(
        self,
        api_key: str,
        cache_dir: str | None = str(pathlib.Path("~/.cache/umls/").expanduser()),
    ):
        self._client = None
        self.api_key = api_key
        self.cache = FileCache(serializer=PickleSerializer(), basedir=cache_dir)
        self._call_umls = _get_data
        self._call_client = decorators._handle_cached_function(
            _get_data,  # Pass the function to be wrapped
            cache=self.cache,  # Inject the cache instance
            ignore_args=["client", "set_cache_value"],
            cache_condition=lambda x: x.status_code == 200,  # Condition for caching
        )

    @property
    def client(self) -> httpx.AsyncClient:
        """Return the client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url="https://uts-ws.nlm.nih.gov/rest",
                params={"apiKey": self.api_key},
            )
        return self._client

    def __getstate__(self) -> object:
        """Return the state."""
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Set the state."""
        self.__dict__.update(state)

    def convert_to_cui(self, results: list[dict[str, str]]) -> list[CUI]:
        return [CUI(**result) for result in results]

    def search(
        self,
        query: str,
        endpoint: str = "/search/current",
        k: int = 1,
        retry_fn_constructor: RetryingConstructor = get_default_retry,
    ) -> list[CUI]:
        request = {"string": query}
        results = retry_fn_constructor()(self._call_umls)(self.client, endpoint, request, k)
        return self.convert_to_cui(results)

    def synonyms(
        self,
        cui: str,
        language: UmlsLanguages = "ENG",
        retry_fn_constructor: RetryingConstructor = get_default_retry,
    ) -> list[str]:
        endpoint = f"/content/current/CUI/{cui}/atoms"
        request = {
            "language": language,
        }
        results = retry_fn_constructor()(self._call_umls)(self.client, endpoint, request)
        synonyms = set()
        for result in results:
            synonyms.add(result.get("name").lower())
        return list(synonyms)
