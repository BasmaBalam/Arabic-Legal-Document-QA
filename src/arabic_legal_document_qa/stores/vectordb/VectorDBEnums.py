from enum import Enum
from qdrant_client.http.models import Distance


class VectorDBEnums(Enum):
    QDRANT = "QDRANT"

class DistanceMethodEnums(Enum):
    COSINE = Distance.COSINE
    DOT = Distance.DOT
    EUCLID = Distance.EUCLID
    MANHATTAN = Distance.MANHATTAN