"""
Memory Search Module
--------------------
负责记忆的检索（Recall）。
核心难点在于如何根据用户模糊的输入，精确召回相关记忆。
"""

from ..utils.logger import get_logger
from .graph_retriever import GraphRetriever
from ..llm.prompt_manager import PromptManager
from ..llm.llm_module import LLMModule
from typing import Tuple, Dict, List, Any
from ..types import KnowledgeItem, GraphEntityType
from ..database.database_service import get_knowledge_from_buffer, VectorStore, KnowledgeGraph, write_knowledge_buffers, write_used_memory_uuid
import asyncio

from sqlalchemy.orm import Session
from redis import Redis


class MemorySearcher:
    def __init__(self, config: Dict[str, Any], prompt_manager: PromptManager):
        self.logger = get_logger(__name__)
        self.config = config
        self.llm = LLMModule(config["llm_module"], prompt_manager)
        self.max_k_vector_entities = config.get("max_k_vector_entities", 3)
        self.max_k_graph_entities = config.get("max_k_graph_entities", 3)

        self.funcname_project_dict = {
            "inherit_memory": self._inherit_memory,
            "v_search": self._vector_search,
            "g_search_song": self._retrieve_one_entity,
            "g_search_song_lyrics": self._get_song_lyrics,
        }
        self.required_database = {
            "inherit_memory": "last_search_results",
            "v_search": "vector_store",
            "g_search_song": "knowledge_db",
            "g_search_song_lyrics": "knowledge_db",
        }

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

        database_dict = {
            "last_search_results": {"last_search_results": last_search_results},
            "vector_store": {"vector_store": vector_store, "used_uuid": used_uuid, "user_id": user_id},
            "knowledge_db": {"knowledge_db": knowledge_db},
        }

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
            if funcname not in self.funcname_project_dict:
                self.logger.warning(f"Unknown search function: {funcname}")
                continue
            search_func = self.funcname_project_dict[funcname]
            try:
                result = await search_func(
                    **database_dict[self.required_database[funcname]],
                    **kwargs,
                )
                if isinstance(result, list):
                    returned_results.extend(result)
                else:
                    returned_results.append(result)
            except Exception as e:
                self.logger.error(f"Error executing {funcname} with args {kwargs}: {e}")

        returned_results = duplicate_removal(returned_results)
        await asyncio.to_thread(write_knowledge_buffers, db, redis, user_id, returned_results) # 将本次提取的记忆保存到redis和db
        await asyncio.to_thread(write_used_memory_uuid, redis, user_id, used_uuid) # 将本次读记忆的uuid保存到redis，马上会在写记忆用到
        return returned_results

    async def _generate_search_queries(self, user_input: str, history: str, last_search_results: List[str]) -> List[Tuple[str, Dict[str, str]]]:
        """
        使用LLM分析用户意图，生成搜索查询。
        这是提高召回率的关键步骤：将"记得那首歌吗"转换为"用户上次提到的歌曲"。
        """

        cmd: List[Tuple[str, Dict[str, str]]] = []
        last_search_results_str = "\n".join([f"{idx}. {content[:100]}" for idx, content in enumerate(last_search_results)])
        try:
            response: str = await self.llm.generate_response(
                    user_input=user_input,
                    history=history,
                    last_search_results=last_search_results_str,
                    max_k_graph_entities=self.max_k_graph_entities,
                    max_k_vector_entities=self.max_k_vector_entities,
                )
            
            response = response.split("\n")
            self.logger.debug(f"Generated search queries: {response} for input: {user_input}")

            # 解析LLM输出，构建搜索命令列表
            for line in response:
                if line.startswith("##"):
                    break
                if line == "":
                    continue
                if "(" not in line or ")" not in line:
                    self.logger.warning(f"Unrecognized command format: {line}")
                    continue
                funcname, args_str = line.split("(", 1)
                args_str = args_str.rstrip(")")
                kwargs = {}
                args = args_str.split(",")

                for arg in args:
                    if "=" in arg and "[" in arg and "]" not in arg:  # list argument split by comma
                        key, value = arg.split("=", 1)
                        list_values = value.strip().lstrip("[")
                        for next_arg in args[args.index(arg) + 1 :]:
                            if "]" in next_arg:
                                list_values += "," + next_arg.strip().rstrip("]")
                                break
                            else:
                                list_values += "," + next_arg.strip()
                        kwargs[key.strip()] = list_values
                    elif "=" in arg:
                        key, value = arg.split("=", 1)
                        kwargs[key.strip()] = value.strip().strip("'\"[]")

                cmd.append((funcname.strip(), kwargs))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.error(f"Error generating search queries: {e}")
        finally:
            return cmd

    async def _inherit_memory(self,last_search_results: List[str], content_ids: str) -> List[str]:
        """
        继承之前检索到的记忆内容
        """
        results = []
        try:
            ids = [int(cid.strip()) for cid in content_ids.split(",")]
            for idx in ids:
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

    async def _get_song_lyrics(self,knowledge_graph: KnowledgeGraph,  song_name: str) -> str:
        """
        根据歌曲名称检索歌词
        """
        song_name = song_name.strip("'\"《》").strip()
        entity = self.graph_retriever.retrieve_one_entity(knowledge_graph, song_name)
        if entity and entity.entity_type == GraphEntityType.SONG and entity.properties.get("lyrics", ""):
            return f"{song_name}的歌词:\n{entity.properties.get('lyrics', '')}"
        elif entity and entity.entity_type != GraphEntityType.SONG:
            return f"找到名为《{song_name}》的实体，但它不是一首歌曲。"

        return f"未找到《{song_name}》的歌词信息。"

    async def _retrieve_one_entity(self, knowledge_graph: KnowledgeGraph, entity_name: str) -> str:
        """
        根据实体名称检索单个实体
        """
        entity_name = entity_name.strip("'\"《》").strip()
        if entity_name == "洛天依":
            return ""  # 不要检索自己啊

        entity = self.graph_retriever.retrieve_one_entity(knowledge_graph, entity_name)
        if entity and entity.properties.get("summary", ""):
            return f"{entity.name}的简介: {entity.properties.get('summary', '')}"

        return f"未找到关于{entity_name}的相关信息。"

    async def _get_neighbors(self, knowledge_graph: KnowledgeGraph, entity_name: str, neighbor_type: str) -> List[str]:
        """
        获取指定实体的邻居节点
        """
        neighbors = self.graph_retriever.get_neighbors(
            knowledge_graph, entity_name, neighbor_type=neighbor_type, needed_neighbors=self.max_k_graph_entities
        )
        ret = []
        for neighbor, _ in neighbors:
            ret.append(f"{neighbor.name}: {neighbor.properties.get('summary', '')}")
        return ret

    async def _get_shared_neighbors(self, knowledge_graph: KnowledgeGraph, entity_name1: str, entity_name2: str, neighbor_type: str) -> List[str]:
        """
        获取两个实体的共同邻居节点
        """
        shared_neighbors = self.graph_retriever.get_shared_neighbors(
            knowledge_graph,
            entity_name1,
            entity_name2,
            neighbor_type=neighbor_type,
            needed_neighbors=self.max_k_graph_entities,
        )
        ret = []
        for neighbor in shared_neighbors:
            ret.append(f"{neighbor.name}: {neighbor.properties.get('summary', '')}")
        return ret

    async def _find_connections(self, knowledge_graph: KnowledgeGraph, entity_name1: str, entity_name2: str) -> List[str]:
        """
        查找两个实体之间的连接路径
        """
        connections =  self.graph_retriever.find_connections(
            knowledge_graph, entity_name1, entity_name2, needed_path_num=self.max_k_graph_entities
        )
        return connections
