from urllib.parse import urljoin

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    modelbox_url: str = "http://0.0.0.0:8228"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def modelbox_info_url(self) -> str:
        return urljoin(self.modelbox_url, "info")

    @property
    def modelbox_file_url(self) -> str:
        return urljoin(self.modelbox_url, "file")

    @property
    def modelbox_media_model_url(self) -> str:
        return urljoin(self.modelbox_url, "media_model")


settings = Settings()
