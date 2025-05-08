# -*- coding: utf-8 -*-
#

import asyncio
import os
import json
from ten import (
    Extension,
    TenEnv,
    Cmd,
    Data,
    StatusCode,
    CmdResult,
)

import threading
from datetime import datetime


class AliPGDBExtension(Extension):
    def __init__(self, name):
        self.stopEvent = asyncio.Event()
        self.thread = None
        self.loop = None
        self.access_key_id = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID")
        self.access_key_secret = os.environ.get(
            "ALIBABA_CLOUD_ACCESS_KEY_SECRET"
        )
        self.region_id = os.environ.get("ADBPG_INSTANCE_REGION")
        self.dbinstance_id = os.environ.get("ADBPG_INSTANCE_ID")
        self.endpoint = "gpdb.aliyuncs.com"
        self.model = None
        self.account = os.environ.get("ADBPG_ACCOUNT")
        self.account_password = os.environ.get("ADBPG_ACCOUNT_PASSWORD")
        self.namespace = os.environ.get("ADBPG_NAMESPACE")
        self.namespace_password = os.environ.get("ADBPG_NAMESPACE_PASSWORD")

    async def __thread_routine(self, ten_env: TenEnv):
        ten_env.log_info("__thread_routine start")
        self.loop = asyncio.get_running_loop()
        ten_env.on_start_done()
        await self.stopEvent.wait()

    async def stop_thread(self):
        self.stopEvent.set()

    def on_start(self, ten: TenEnv) -> None:
        ten.log_info("on_start")
        self.access_key_id = self.get_property_string(
            ten, "ALIBABA_CLOUD_ACCESS_KEY_ID", self.access_key_id
        )
        self.access_key_secret = self.get_property_string(
            ten, "ALIBABA_CLOUD_ACCESS_KEY_SECRET", self.access_key_secret
        )
        self.region_id = self.get_property_string(
            ten, "ADBPG_INSTANCE_REGION", self.region_id
        )
        self.dbinstance_id = self.get_property_string(
            ten, "ADBPG_INSTANCE_ID", self.dbinstance_id
        )
        self.account = self.get_property_string(
            ten, "ADBPG_ACCOUNT", self.account
        )
        self.account_password = self.get_property_string(
            ten, "ADBPG_ACCOUNT_PASSWORD", self.account_password
        )
        self.namespace = self.get_property_string(
            ten, "ADBPG_NAMESPACE", self.namespace
        )
        self.namespace_password = self.get_property_string(
            ten, "ADBPG_NAMESPACE_PASSWORD", self.namespace_password
        )

        if self.region_id in (
            "cn-beijing",
            "cn-hangzhou",
            "cn-shanghai",
            "cn-shenzhen",
            "cn-hongkong",
            "ap-southeast-1",
            "cn-hangzhou-finance",
            "cn-shanghai-finance-1",
            "cn-shenzhen-finance-1",
            "cn-beijing-finance-1",
        ):
            self.endpoint = "gpdb.aliyuncs.com"
        else:
            self.endpoint = f"gpdb.{self.region_id}.aliyuncs.com"

        # lazy import packages which requires long time to load
        from .client import AliGPDBClient
        from .model import Model

        client = AliGPDBClient(
            ten, self.access_key_id, self.access_key_secret, self.endpoint
        )
        self.model = Model(ten, self.region_id, self.dbinstance_id, client)
        self.thread = threading.Thread(
            target=asyncio.run, args=(self.__thread_routine(ten),)
        )

        # Then 'on_start_done' will be called in the thread
        self.thread.start()
        return

    def on_stop(self, ten: TenEnv) -> None:
        ten.log_info("on_stop")
        if self.thread is not None and self.thread.is_alive():
            asyncio.run_coroutine_threadsafe(self.stop_thread(), self.loop)
            self.thread.join()
        self.thread = None
        ten.on_stop_done()
        return

    def on_data(self, ten: TenEnv, data: Data) -> None:
        pass

    def on_cmd(self, ten: TenEnv, cmd: Cmd) -> None:
        try:
            cmd_name = cmd.get_name()
            ten.log_info(f"on_cmd [{cmd_name}]")
            if cmd_name == "create_collection":
                asyncio.run_coroutine_threadsafe(
                    self.async_create_collection(ten, cmd), self.loop
                )
            elif cmd_name == "delete_collection":
                asyncio.run_coroutine_threadsafe(
                    self.async_delete_collection(ten, cmd), self.loop
                )
            elif cmd_name == "upsert_vector":
                asyncio.run_coroutine_threadsafe(
                    self.async_upsert_vector(ten, cmd), self.loop
                )
            elif cmd_name == "query_vector":
                asyncio.run_coroutine_threadsafe(
                    self.async_query_vector(ten, cmd), self.loop
                )
            else:
                ten.return_result(CmdResult.create(StatusCode.ERROR), cmd)
        except Exception:
            ten.return_result(CmdResult.create(StatusCode.ERROR), cmd)

    async def async_create_collection(self, ten: TenEnv, cmd: Cmd):
        collection = cmd.get_property_string("collection_name")
        dimension = 1024
        try:
            dimension = cmd.get_property_int("dimension")
        except Exception as e:
            ten.log_warn(f"Error: {e}")

        err = await self.model.create_collection_async(
            self.account, self.account_password, self.namespace, collection
        )
        if err is None:
            await self.model.create_vector_index_async(
                self.account,
                self.account_password,
                self.namespace,
                collection,
                dimension,
            )
            ten.return_result(CmdResult.create(StatusCode.OK), cmd)
        else:
            ten.return_result(CmdResult.create(StatusCode.ERROR), cmd)

    async def async_upsert_vector(self, ten: TenEnv, cmd: Cmd):
        start_time = datetime.now()
        collection = cmd.get_property_string("collection_name")
        file = cmd.get_property_string("file_name")
        content = cmd.get_property_string("content")
        obj = json.loads(content)
        rows = [(file, item["text"], item["embedding"]) for item in obj]

        err = await self.model.upsert_collection_data_async(
            collection, self.namespace, self.namespace_password, rows
        )
        ten.log_info(
            f"upsert_vector finished for file {file}, collection {collection}, rows len {len(rows)}, err {err}, cost {int((datetime.now() - start_time).total_seconds() * 1000)}ms"
        )
        if err is None:
            ten.return_result(CmdResult.create(StatusCode.OK), cmd)
        else:
            ten.return_result(CmdResult.create(StatusCode.ERROR), cmd)

    async def async_query_vector(self, ten: TenEnv, cmd: Cmd):
        start_time = datetime.now()
        collection = cmd.get_property_string("collection_name")
        embedding = cmd.get_property_to_json("embedding")
        top_k = cmd.get_property_int("top_k")
        vector = json.loads(embedding)
        response, error = await self.model.query_collection_data_async(
            collection,
            self.namespace,
            self.namespace_password,
            vector,
            top_k=top_k,
        )
        ten.log_info(
            f"query_vector finished for collection {collection}, embedding len {len(embedding)}, err {error}, cost {int((datetime.now() - start_time).total_seconds() * 1000)}ms"
        )

        if error:
            return ten.return_result(CmdResult.create(StatusCode.ERROR), cmd)
        else:
            body = self.model.parse_collection_data(response.body)
            ret = CmdResult.create(StatusCode.OK)
            ret.set_property_from_json("response", body)
            ten.return_result(ret, cmd)

    async def async_delete_collection(self, ten: TenEnv, cmd: Cmd):
        collection = cmd.get_property_string("collection_name")
        # pylint: disable=too-many-function-args
        err = await self.model.delete_collection_async(
            self.account, self.account_password, self.namespace, collection
        )
        if err is None:
            return ten.return_result(CmdResult.create(StatusCode.OK), cmd)
        else:
            return ten.return_result(CmdResult.create(StatusCode.ERROR), cmd)

    def get_property_string(self, ten: TenEnv, key: str, default: str) -> str:
        try:
            return ten.get_property_string(key.lower())
        except Exception as e:
            ten.log_error(f"Error: {e}")
            return default
