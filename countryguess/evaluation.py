"""Load user drawings from the drawingstore for model evaluation."""

import logging

import requests
from shapely.geometry import shape as geometry_shape

from .data import ReferenceDataset, geom_to_img
from .utils import normalize_geom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DrawingStoreClient:
    """Read drawings through the drawingstore."""

    def __init__(self, base_url: str, timeout: float = 30) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def list_drawings(
        self,
        *,
        validated: bool | None = None,
        exclude_other: bool = False,
        page_size: int = 1000,
    ) -> list[dict]:
        """Return all drawings matching the generic collection filters.

        The API is paginated, so this method follows all pages and returns an
        in-memory snapshot for one evaluation run.
        """
        drawings: list[dict] = []
        offset = 0

        while True:
            params: dict[str, str | int] = {"limit": page_size, "offset": offset}
            if validated is not None:
                params["validated"] = str(validated).lower()
            if exclude_other:
                params["exclude_other"] = "true"

            try:
                response = requests.get(
                    f"{self.base_url}/drawings",
                    params=params,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as error:
                raise RuntimeError(
                    f"failed to fetch drawings from {self.base_url}: {error}"
                ) from error

            page = payload.get("drawings") if isinstance(payload, dict) else None
            total = payload.get("total") if isinstance(payload, dict) else None
            if not isinstance(page, list) or not isinstance(total, int):
                raise TypeError("drawingstore returned an invalid drawings response")

            drawings.extend(page)
            if not page or len(drawings) >= total:
                return drawings[:total]
            offset += len(page)


class EvaluationDataset(ReferenceDataset):
    """Validated user drawings fetched from the drawingstore for evaluation."""

    def __init__(self, drawing_store_url: str, shape=(64, 64)):
        super().__init__(shape=shape)

        reference_countries = {
            sample["country_name"] for sample in self.reference_samples
        }
        stored_drawings = DrawingStoreClient(drawing_store_url).list_drawings(
            validated=True,
            exclude_other=True,
        )

        self.samples = []
        skipped = 0
        for drawing in stored_drawings:
            country_name = drawing.get("country")
            geometry = drawing.get("geometry")
            if country_name not in reference_countries or not isinstance(
                geometry, dict
            ):
                skipped += 1
                continue
            self.samples.append(
                {
                    "country_name": country_name,
                    "geometry": normalize_geom(
                        geometry_shape(geometry), shape=self.shape
                    ),
                }
            )

        logger.info(
            "Loaded %d evaluation drawings from drawingstore (%d skipped)",
            len(self.samples),
            skipped,
        )

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        drawing = geom_to_img(item["geometry"], self.shape)

        return {"country_name": item["country_name"], "drawing": drawing}
