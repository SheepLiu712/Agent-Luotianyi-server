"""
Memory Search Module
--------------------
负责记忆的检索（Recall）。
核心难点在于如何根据用户模糊的输入，精确召回相关记忆。
"""

from ..utils.logger import get_logger
from ..music.knowledge_service import get_song_introduction, get_song_lyrics
from ..llm.prompt_manager import PromptManager
from ..llm.llm_module import LLMModule
from typing import Tuple, Dict, List, Any
from ..types import KnowledgeItem, GraphEntityType
from ..types.tool_type import MyTool, ToolFunction, ToolOneParameter
from ..database.database_service import get_knowledge_from_buffer, VectorStore, KnowledgeGraph, write_knowledge_buffers, write_used_memory_uuid
import asyncio 
import json

from sqlalchemy.orm import Session
from redis import Redis



class MemorySearcher:
    def __init__(self, config: Dict[str, Any], prompt_manager: PromptManager):
        self.logger = get_logger(__name__)
        self.config = config
        self.llm = LLMModule(config["llm_module"], prompt_manager)
        self.max_k_vector_entities = config.get("max_k_vector_entities", 3)
        self.max_k_graph_entities = config.get("max_k_graph_entities", 3)

        # 设置工具列表
        self.tools: List[MyTool] = []
        self.tool_map: Dict[str, MyTool] = {}
        self.set_up_tools()

    async def search(
        self,
        db: Session,
        redis: Redis,
        vector_store: VectorStore,
        knowledge_db: Session,
        user_id: str,
        user_input: str,
        history: str,
    ) -> List[str]:
        """
        执行混合检索策略
        """
        # 1. 查询理解与扩展 (Query Expansion)
        # 利用LLM将用户的自然语言转换为具体的搜索意图
        # 获取暂存的上次搜索结果
        last_search_results: List[str] = await asyncio.to_thread(get_knowledge_from_buffer, db, redis, user_id)
        search_queries = await self._generate_search_queries(user_input, history, last_search_results)
        used_uuid = set()

        # 构建工具映射

        # 构建上下文映射,用于注入 additional_required_params
        context_map = {}
        context_map["last_search_results"] = last_search_results
        context_map["vector_store"] = vector_store
        context_map["used_uuid"] = used_uuid
        context_map["user_id"] = user_id
        context_map["knowledge_db"] = knowledge_db

        def duplicate_removal(seq: List[str]) -> List[str]:
            """移除列表中的重复项，保持顺序不变"""
            seen = set()
            result = []
            for item in seq:
                item_repr = item.strip()[:50]  # 只考虑前50个字符以判断重复
                if item_repr not in seen:
                    seen.add(item_repr)
                    result.append(item)
            return result

        returned_results = []
        for funcname, kwargs in search_queries:
            if funcname not in self.tool_map:
                self.logger.warning(f"Unknown search function: {funcname}")
                continue
            
            tool = self.tool_map[funcname]
            
            # 准备运行时参数
            call_kwargs = {}
            call_kwargs.update(kwargs) # LLM提供的参数
            
            # 注入额外参数
            if tool.additional_required_params:
                for req_param in tool.additional_required_params:
                    if req_param in context_map:
                        call_kwargs[req_param] = context_map[req_param]
                    else:
                        self.logger.warning(f"Missing context parameter {req_param} for tool {funcname}")

            try:
                result = await tool.tool_func(**call_kwargs)
                if isinstance(result, list):
                    returned_results.extend(result)
                else:
                    returned_results.append(result)
            except Exception as e:
                self.logger.error(f"Error executing {funcname} with args {kwargs}: {e}")
                import traceback
                traceback.print_exc()

        returned_results = duplicate_removal(returned_results)
        await asyncio.to_thread(write_knowledge_buffers, db, redis, user_id, returned_results) # 将本次提取的记忆保存到redis和db
        await asyncio.to_thread(write_used_memory_uuid, redis, user_id, used_uuid) # 将本次读记忆的uuid保存到redis，马上会在写记忆用到
        return returned_results

    async def _generate_search_queries(self, user_input: str, history: str, last_search_results: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        使用LLM分析用户意图，生成搜索查询。
        """
        cmd: List[Tuple[str, Dict[str, Any]]] = []
        last_search_results_str = "\n".join([f"{idx}. {content[:100]}" for idx, content in enumerate(last_search_results)])
        
        tools_str = self.tool_to_str()
        
        try:
            response_str: str = await self.llm.generate_response(
                    user_input=user_input,
                    history=history,
                    last_search_results=last_search_results_str,
                    tools=tools_str,
                    use_json=True
                )
            
            self.logger.debug(f"Generated search queries raw: {response_str}")

            # extract json from potential markdown code blocks
            json_str = response_str
            if "```json" in response_str:
                json_str = response_str.split("```json")[1].split("```")[0].strip()
            elif "```" in response_str:
                json_str = response_str.split("```")[1].split("```")[0].strip()
            
            try:
                # remove potential non-json leading/trailing text if needed
                start_idx = json_str.find("{")
                end_idx = json_str.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    json_str = json_str[start_idx:end_idx+1]

                response_data = json.loads(json_str)
                tool_uses = response_data.get("tool_use", [])
                
                if isinstance(tool_uses, list):
                    for tool_use in tool_uses:
                        tool_name = tool_use.get("tool_name")
                        parameters = tool_use.get("parameters", {})
                        if tool_name:
                            cmd.append((tool_name, parameters))
                        
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to decode JSON from LLM response: {json_str}, error: {e}")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.error(f"Error generating search queries: {e}")
        finally:
            return cmd
        
    def register_tools(self, tools: Dict[str, MyTool]):
        """
        注册外部工具
        """
        for name, tool in tools.items():
            self.tool_map[name] = tool

    async def _inherit_memory(self,last_search_results: List[str], content_ids: List[int]) -> List[str]:
        """
        继承之前检索到的记忆内容
        """
        results = []
        try:
            for idx in content_ids:
                if 0 <= idx < len(last_search_results):
                    results.append(last_search_results[idx])
        except Exception as e:
            self.logger.error(f"Error in inherit_memory with content_ids {content_ids}: {e}")
        finally:
            return results

    async def _vector_search(self,vector_store: VectorStore, used_uuid: set, user_id: str, query: str) -> List[str]:
        """
        基于向量检索的记忆搜索
        """
        results = await vector_store.search(user_id, query, k=self.max_k_vector_entities) # 这个一定要异步，因为需要网络请求嵌入
        combined_result = []
        for doc, score in results:
            if score < 0.50:
                break
            if doc.id not in used_uuid:
                used_uuid.add(doc.id)
                timestamp = doc.metadata.get("timestamp", "unknown time")
                combined_result.append(f"在{timestamp}, {doc.get_content()}")
        return combined_result

    async def _search_song_intro(self, knowledge_db: Session, song_name: str) -> str:
        """
        根据歌名查询歌曲介绍
        """
        song_name = song_name.strip("'\"《》").strip()
        introduction = await asyncio.to_thread(get_song_introduction, knowledge_db, song_name)
        
        if introduction:
            return f"《{song_name}》的介绍:\n{introduction}"
        return f"未找到《{song_name}》的相关介绍。"

    async def _search_song_lyrics(self, knowledge_db: Session, song_name: str) -> str:
        """
        根据歌名查询歌词
        """
        song_name = song_name.strip("'\"《》").strip()
        lyrics = await asyncio.to_thread(get_song_lyrics, knowledge_db, song_name)
        
        if lyrics:
            return f"《{song_name}》的歌词:\n{lyrics}"
        return f"未找到《{song_name}》的歌词信息。"

    def set_up_tools(self):

        inherit_memory_tool = MyTool(
            name="inherit_memory",
            description="继承之前检索到的记忆内容",
            tool_func=self._inherit_memory,
            tool_interface= ToolFunction(
                name="inherit_memory",
                description="继承之前检索到的记忆内容",
                parameters=[
                    ToolOneParameter(
                        name="content_ids",
                        type="List[int]",
                        description="之前检索到的记忆内容的编号列表，格式如 [0,1,2]，对应上次搜索结果中的索引",
                    ),
                ],
            ),
            additional_required_params=["last_search_results"]
        )
        self.tool_map[inherit_memory_tool.name] = inherit_memory_tool

        memory_search = MyTool(
            name="memory_search",
            description="检索长期记忆",
            tool_func=self._vector_search,
            tool_interface= ToolFunction(
                name="memory_search",
                description="检索长期记忆，当用户提到某个过去的事件、对话或者是模糊的信息时，使用此工具搜索数据库。",
                parameters=[
                    ToolOneParameter(
                        name="query",
                        type="str",
                        description="用于检索的查询语句",
                    ),
                ],
            ),
            additional_required_params=["vector_store", "used_uuid", "user_id"]
        )
        self.tool_map[memory_search.name] = memory_search

        search_song = MyTool(
            name="search_song_intro",
            description="根据歌名查询歌曲介绍",
            tool_func=self._search_song_intro,
            tool_interface= ToolFunction(
                name="search_song_intro",
                description="查询歌曲的详细介绍信息，当用户询问某首歌的信息时使用",
                parameters=[
                    ToolOneParameter(
                        name="song_name",
                        type="str",
                        description="歌曲名称",
                    ),
                ],
            ),
            additional_required_params=["knowledge_db"]
        )
        self.tool_map[search_song.name] = search_song

        search_song_lyrics = MyTool(
            name="search_song_lyrics",
            description="根据歌名查询歌词",
            tool_func=self._search_song_lyrics,
            tool_interface= ToolFunction(
                name="search_song_lyrics",
                description="查询歌曲的歌词内容",
                parameters=[
                    ToolOneParameter(
                        name="song_name",
                        type="str",
                        description="歌曲名称",
                    ),
                ],
            ),
            additional_required_params=["knowledge_db"]
        )
        self.tool_map[search_song_lyrics.name] = search_song_lyrics

    def tool_to_str(self) -> str:
        tool_descriptions = []
        for tool in self.tool_map.values():
            tool_desc = tool.get_interface_str()
            tool_descriptions.append(tool_desc)
        return "\n".join(tool_descriptions)