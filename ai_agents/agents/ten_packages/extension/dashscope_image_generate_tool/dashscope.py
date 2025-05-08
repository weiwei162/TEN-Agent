#
#
# Agora Real Time Engagement
# Created by Wei Hu in 2024-08.
# Copyright (c) 2024 Agora IO. All rights reserved.
#
#
from dataclasses import dataclass
import dashscope

from ten.async_ten_env import AsyncTenEnv
from ten_ai_base.config import BaseConfig


@dataclass
class DashScopeImageGenerateToolConfig(BaseConfig):
    api_key: str = ""
    model: str = "wanx2.1-t2i-turbo"
    size: str = "1024*1024"
    n: int = 1
    prompt_extend: bool = True
    watermark: bool = False

class DashScopeImageGenerateClient:
    def __init__(self, ten_env: AsyncTenEnv, config: DashScopeImageGenerateToolConfig):
        self.config = config
        dashscope.api_key = config.api_key
        ten_env.log_info(f"DashScopeImageGenerateClient initialized with config: {config.api_key}")

    async def generate_images(self, prompt: str) -> str:
        try:
            # 创建任务
            rsp = dashscope.ImageSynthesis.async_call(
                api_key=self.config.api_key,
                model=self.config.model,
                prompt=prompt,
                size=self.config.size,
                n=self.config.n
            )

            if rsp.status_code != 200:
                raise RuntimeError(f"status_code: {rsp.status_code}, code: {rsp.code}, message: {rsp.message}")

            # 等待结果
            rsp = dashscope.ImageSynthesis.wait(rsp)

            if rsp.status_code == 200:
                return rsp.output.results[0].url
            else:
                raise RuntimeError(f"status_code: {rsp.status_code}, code: {rsp.code}, message: {rsp.message}")

        except Exception as e:
            raise RuntimeError(f"GenerateImages failed, err: {e}") from e
