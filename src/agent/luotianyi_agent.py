"""
洛天依Agent主类

实现洛天依角色扮演对话Agent的核心逻辑
"""

from typing import AsyncGenerator, List, Dict, Any, Optional
from abc import ABC, abstractmethod
import time
from sqlalchemy.orm import Session
import redis
import asyncio
import json
from fastapi import UploadFile
import re
import base64

from ..llm.prompt_manager import PromptManager
from .main_chat import MainChat, OneSentenceChat, SongSegmentChat, OneResponseLine
from .planner import Planner
from .conversation_manager import ConversationManager
from ..utils.logger import get_logger
from ..tts import TTSModule
from ..utils.enum_type import ContextType, ConversationSource
from ..memory.memory_manager import MemoryManager
from ..service.types import ChatResponse
from ..music.singing_manager import SingingManager
from ..vision.vision_module import VisionModule


def get_available_expression(config_path: str = "config/live2d_interface_config.json") -> List[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict = json.load(f)
    expressions: Dict = config.get("expression_projection", {})
    return list(expressions.keys())


class LuoTianyiAgent:
    """洛天依Agent类

    实现洛天依角色扮演对话Agent的核心逻辑
    """

    def __init__(self, config: Dict[str, Any], tts_module: TTSModule) -> None:
        """初始化洛天依Agent

        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = get_logger("LuoTianyiAgent")
        self.prompt_manager = PromptManager(self.config.get("prompt_manager", {}))  # 提示管理器

        # 各种模块初始化
        self.conversation_manager = ConversationManager(
            self.config.get("conversation_manager", {}), self.prompt_manager
        )  # 对话管理器
        self.singing_manager = SingingManager(config={})  # 唱歌管理器
        memory_config = self.config.get("memory_manager", {})
        self.memory_manager = MemoryManager(memory_config, self.prompt_manager)  # 记忆管理器
        self.memory_manager.memory_searcher.register_tools(self.singing_manager.get_tools())  # 注册唱歌工具

        self.tts_engine = tts_module  # TTS模块

        self.main_chat = MainChat(
            self.config["main_chat"],
            self.prompt_manager,
            available_tone=tts_module.get_available_tones(),
            available_expression=get_available_expression(),
        )
        self.planner = Planner(config["planner"], self.prompt_manager, self.singing_manager)
        self.vision_module = VisionModule(config["vision_module"], self.prompt_manager)
 
    async def handle_user_text_input(
        self,
        user_id: str,
        text: str,
        db: Session,
        redis: redis.Redis,
        vector_store: Any = None,
        knowledge_db: Session = None,
    ) -> AsyncGenerator[ChatResponse, None]:
        """处理用户文本输入，生成回复（流式）。"""
        self.logger.info(f"Agent handling text input for {user_id}: {text}")
        await self.conversation_manager.add_conversation(
            db, redis, user_id, ConversationSource.USER, text, type=ContextType.TEXT, data=None
        )
        async for response in self.shared_process_pipeline(user_id, text, db, redis, vector_store, knowledge_db):
            yield response


    async def handle_user_pic_input(
        self,
        user_id: str,
        image: UploadFile,
        image_client_path: str,
        db: Session,
        redis: redis.Redis,
        vector_store: Any,
        knowledge_db: Session,
    ) -> AsyncGenerator[ChatResponse, None]:
        """处理用户图片输入，生成回复（流式）。"""
        self.logger.info(f"Agent handling picture input for {user_id}")
        # 将图片通过vlm模块转换为描述文本
        image_bytes = await image.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_description = await self.vision_module.describe_image(image_base64)  # TODO: 实现视觉模块
        await self.conversation_manager.add_conversation(
            db,
            redis,
            user_id,
            ConversationSource.USER,
            content=image_description,
            type=ContextType.PICTURE,
            data={"image_client_path": image_client_path, "image_bytes": image_bytes},
        )
        async for response in self.shared_process_pipeline(user_id, image_description, db, redis, vector_store, knowledge_db):
            yield response

    async def shared_process_pipeline(
        self,
        user_id: str,
        text: str,
        db: Session,
        redis: redis.Redis,
        vector_store: Any = None,
        knowledge_db: Session = None,
    ) -> AsyncGenerator[ChatResponse, None]:
        """
        处理用户输入，生成回复（流式）。
        逻辑对应原 luotianyi_agent.py 中的 handle_user_input，但改为无状态和异步。

        Args:
            user_id: 用户ID，用于区分不同用户的上下文
            text: 用户输入的文本
            db: SQL数据库会话，用于持久化存储
            redis: Redis客户端，用于快速存取短期记忆/上下文
            vector_store: 向量数据库接口
            knowledge_db: Session = None,

        Yields:
            ChatResponse: 包含文本、表情、语音等信息的响应片段
        """
        # 1. 获取上下文 (Cache Aside 模式：优先查Redis，未命中则查DB并回写)
        context_data = await self.conversation_manager.get_context(db, redis, user_id)

        # 2. 记忆与知识检索 (并发执行，避免阻塞)
        username_task = asyncio.create_task(self.memory_manager.get_username(db, redis, user_id))
        knowledge_task = asyncio.create_task(
            self.memory_manager.get_knowledge(db, redis, vector_store, knowledge_db, user_id, text, context_data)
        )
        user_nickname, retrieved_knowledge = await asyncio.gather(username_task, knowledge_task)

        # 3. 调用LLM生成文本 (异步流式或分段)
        # 3a. 调用规划器生成行动计划
        planning_step = await self.planner.generate_response(
            user_input=text, conversation_history=context_data, retrieved_knowledge=retrieved_knowledge
        )
        # 3b. 根据行动计划调用主聊天模块生成回复
        llm_responses = await self.main_chat.generate_response(
            user_input=text,
            conversation_history=context_data,
            retrieved_knowledge=retrieved_knowledge,
            username=user_nickname,
            planning_step=planning_step,
        )

        # 4. 记忆写入（异步）
        agent_response_contents = [resp.get_content() for resp in llm_responses]
        write_memory_task = asyncio.create_task(
            self.memory_manager.post_process_interaction(
                db=db,
                redis=redis,
                vector_store=vector_store,
                user_id=user_id,
                user_input=text,
                agent_response_content=agent_response_contents,
                history=context_data,
            )
        )

        # 5. 处理生成的文本片段
        for resp in llm_responses:
            if resp.type == ContextType.SING:
                # 处理唱歌片段
                parameter: SongSegmentChat = resp.parameters
                self.logger.info(f"Singing segment for {user_id}: {parameter.song} - {parameter.segment}")
                lrc_lines, sing_audio_base64_str = self.singing_manager.get_song_segment(parameter.song, parameter.segment)
                lrc = [line.content for line in lrc_lines]
                sent_text = "（唱歌）：《" + parameter.song + "》\n" + "\n".join(lrc)
                await self.conversation_manager.add_conversation(
                    db, redis, user_id, ConversationSource.AGENT, sent_text, type=ContextType.SING
                )

                total_length = len(sing_audio_base64_str) if sing_audio_base64_str else 0
                self.logger.info(f"Audio base64 length: {total_length}")

                # 将Base64字符串分块发送，以便客户端可以边接收边缓冲播放
                chunk_size = 1024 * 1024  # 1MB per chunk

                if total_length == 0:
                    yield ChatResponse(text=sent_text, expression="唱歌", audio="")
                else:
                    for i in range(0, total_length, chunk_size):
                        chunk = sing_audio_base64_str[i : i + chunk_size]
                        # 第一帧包含文本和表情，后续帧只包含音频数据
                        if i == 0:
                            yield ChatResponse(
                                text=sent_text, expression="唱歌", audio=chunk, is_final_package=(i + chunk_size >= total_length)
                            )
                        else:
                            yield ChatResponse(
                                text="",  # 后续帧不重复发送文本
                                expression=None,
                                audio=chunk,
                                is_final_package=(i + chunk_size >= total_length),
                            )
                        # 让出控制权，确保数据能及时通过网络栈发送出去，而不是堆积在缓冲区
                        await asyncio.sleep(0)

            elif resp.type == ContextType.TEXT:
                # 处理普通文本片段
                resp: OneSentenceChat = resp.parameters
                resp_split = self._split_responses([resp])
                for split_resp in resp_split:
                    sent_content = split_resp.content
                    expression = split_resp.expression
                    tone = split_resp.tone if split_resp.tone else "normal"

                    async def fake_tts():
                        return b""

                    tts_task = asyncio.create_task(self.tts_engine.synthesize_speech_with_tone(sent_content, tone))
                    # tts_task = asyncio.create_task(fake_tts())
                    db_task = asyncio.create_task(
                        self.conversation_manager.add_conversation(
                            db, redis, user_id, ConversationSource.AGENT, sent_content, type=ContextType.TEXT
                        )
                    )

                    audio_wav_bytes, _ = await asyncio.gather(tts_task, db_task)
                    audio_data_base64 = self.tts_engine.encode_audio_to_base64(audio_wav_bytes)

                    response = ChatResponse(
                        text=sent_content, expression=expression, audio=audio_data_base64, is_final_package=True
                    )

                    yield response

        # 6. 触发后台任务：长期记忆存储 (不阻塞当前响应)
        # 保存用户输入和 Agent 回复到数据库和向量库
        await write_memory_task
        self.logger.info(f"Finished handling input for {user_id}")

    def _split_responses(self, responses: List[OneSentenceChat]) -> List[OneSentenceChat]:
        """将长文本拆分为多个响应片段

        Args:
            responses: 原始响应列表

        Returns:
            拆分后的响应列表
        """

        def clean_sound_content(text: str) -> str:
            # Remove content within parentheses (Chinese and English)
            return re.sub(r"（.*?）|\(.*?\)", "", text)

        punct_pattern = re.compile(r"^(?:\.{3}|[。，！？~,])+$")

        split_responses: List[OneSentenceChat] = []
        for resp in responses:
            # 使用捕获组 () 保留分隔符
            parts = re.split(r"((?:\.{3}|[。，！？~,]))", resp.content)

            # 获取带标点符号的各个句子
            sentences_with_punct = []
            for s in parts:
                if not s:
                    continue
                if punct_pattern.match(s) and sentences_with_punct:
                    sentences_with_punct[-1] += s
                else:
                    sentences_with_punct.append(s)

            # 只要一句话超过了5个字，就拆分。否则分给下一句一起拆分
            sentence_buffer: str = ""

            for i, sentence in enumerate(sentences_with_punct):
                # check if sentence starts with parenthesis (action/mood)
                match = re.match(r"^(\（.*?\）|\(.*?\))", sentence)
                paren_content = None
                if match:
                    paren_content = match.group(1)
                    sentence = sentence[len(paren_content) :]  # remove from current sentence

                if paren_content:
                    # assign to previous sentence
                    if sentence_buffer.strip():
                        # append to current buffer
                        sentence_buffer += paren_content
                    elif split_responses:
                        # append to last existing response content
                        # No need to update sound_content as it strips parentheses anyway
                        split_responses[-1].content += paren_content
                    else:
                        # no previous sentence, keep it at start
                        sentence = paren_content + sentence

                sentence_buffer += sentence

                # Standard flush condition
                if len(sentence_buffer) >= 6 or i == len(sentences_with_punct) - 1:
                    if sentence_buffer.strip():
                        final_content = sentence_buffer.strip()
                        split_responses.append(
                            OneSentenceChat(
                                content=final_content,
                                expression=resp.expression,
                                tone=resp.tone,
                                sound_content=clean_sound_content(final_content),
                            )
                        )
                        sentence_buffer = ""
        return split_responses

    async def handle_history_request(self, user_id: str, count: int, end_index: int, db: Session) -> Dict[str, Any]:
        """处理历史记录请求

        Args:
            count: 请求的数量
            end_index: 结束索引（不包含），-1表示从最新开始

        Returns:
            (history_list, start_index)
        """

        total_count = await self.conversation_manager.get_total_conversation_count(db, user_id)

        if end_index == -1 or end_index > total_count:
            end_index = total_count

        start_index = max(0, end_index - count)

        # 如果请求范围无效（例如已经到了最开始），返回空列表
        if start_index >= end_index:
            return {"history": [], "start_index": 0}

        history_items = await self.conversation_manager.get_history(db, user_id, start_index, end_index)

        # 转换为UI需要的格式
        ret = {"history": [], "start_index": 0}

        for item in history_items:
            if item.type == ContextType.PICTURE and item.data:
                # 图片消息，返回图片路径
                image_client_path = item.data.get("image_client_path")
                image_server_path = item.data.get("image_server_path")
                content = {"image_client_path": image_client_path, "image_server_path": image_server_path}
                content = json.dumps(content)  # 转换为字符串格式，前端解析后使用
            else:
                content = item.content
            ret["history"].append(
                {"content": content, "source": item.source, "timestamp": item.timestamp, "type": item.type}
            )

        return ret


agent = None


def init_luotianyi_agent(config: Dict[str, Any], tts_module: TTSModule):
    """初始化洛天依Agent实例

    Args:
        config: 配置字典
    Returns:
        LuoTianyiAgent实例
    """
    global agent
    agent = LuoTianyiAgent(config, tts_module)


def get_luotianyi_agent() -> LuoTianyiAgent:
    """获取洛天依Agent实例

    Returns:
        LuoTianyiAgent实例
    """
    if agent is None:
        raise ValueError("LuoTianyiAgent has not been initialized.")
    return agent
