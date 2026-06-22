from nanovllm.config import Config
from nanovllm.engine.cache_connector.base import KVConnectorBase, KVConnectorRole


class CPUCacheConnector(KVConnectorBase):
    def __init__(self, config: Config, role: KVConnectorRole):
        super().__init__(config, role)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        pass

    def clear_connector_metadata(self) -> None:
        pass

    def _get_connector_metadata(self) -> KVConnectorMetadata:
        pass
