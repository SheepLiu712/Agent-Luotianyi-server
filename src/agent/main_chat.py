from ..llm.llm_module import LLMModule
from ..llm.prompt_manager import PromptManager
from typing import Dict, Any, List, Optional
from jinja2 import Template
import time
import dataclasses
import json
from .planner import PlanningStep
from ..utils.logger import get_logger
from ..utils.enum_type import ContextType

@dataclasses.dataclass
class OneResponseLine:
    type: ContextType  # 'say' 或 'sing'
    parameters: Any # SongSegment 或 OneSentenceChat
    def get_content(self) -> str:
        if self.type == ContextType.TEXT:
            return self.parameters.content
        else:
            return f"唱了{self.parameters.song}的选段{self.parameters.segment}"

@dataclasses.dataclass
class SongSegmentChat:
    song: str
    segment: str
    lyrics: str = ""

@dataclasses.dataclass
class OneSentenceChat:
    expression: str
    tone: str
    content: str
    sound_content: str = ""

you_should_dict = {
    "闲聊回复": "简短地回复几句，保持对话的连续性和互动性。回复中不能包含'sing'类型的内容。",
    "认真回复": "认真且详细地回复用户的最新对话内容。回复中不能包含'sing'类型的内容。",
    "唱歌回复": "你需要在回复中包含'sing'类型的内容，唱歌给用户听。",
}

class MainChat:
    def __init__(
        self, config: Dict[str, Any], prompt_manager: PromptManager, available_tone: List[str], available_expression: List[str]
    ) -> None:
        self.logger = get_logger(__name__)
        self.config = config
        self.llm = LLMModule(config["llm_module"], prompt_manager)
        self.variables: List[str] = self.llm.prompt_template.get_variables()

        # 假设 init_static_variables 是用来读取配置文件的，应该是同步的比较好。
        self._init_static_variables_sync(available_tone, available_expression)

    async def generate_response(
        self,
        user_input: str,
        planning_step: PlanningStep,
        conversation_history: str = "",
        retrieved_knowledge: List[str] = "",
        username: str = "",
    ) -> list[OneResponseLine]:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        persona = self.persona
        response_requirements = self.response_requirements
        response_format = self.response_format

        action_desc = planning_step.description
        action_str = action_desc + you_should_dict.get(planning_step.action, "回复中不能包含'sing'类型的内容。")

        # 调用异步的 LLM 生成接口
        response = await self.llm.generate_response(
            user_message=user_input,
            current_time=current_time,
            persona=persona,
            action = action_str,
            response_requirements=response_requirements,
            response_format=response_format,
            conversation_history=conversation_history,
            knowledge="\n".join(retrieved_knowledge),
            username=username,
            use_json=True
        )
        self.logger.debug(f"MainChat LLM response: {response}")

        # 解析 LLM 返回的 JSON 格式响应
        response_list = json.loads(response)
        response_list = response_list.get("response", [])
        result: list[OneResponseLine] = []
        for item in response_list:
            if item.get("type") == "say":
                response_line = OneResponseLine(
                    type=ContextType.TEXT,
                    parameters=OneSentenceChat(
                        expression=item["parameters"].get("expression", ""),
                        tone=item["parameters"].get("tone", ""),
                        content=item["parameters"].get("content", ""),
                    )
                )
            elif item.get("type") == "sing":
                response_line = OneResponseLine(
                    type=ContextType.SING,
                    parameters=SongSegmentChat(
                        song=item["parameters"].get("song", ""),
                        segment=item["parameters"].get("segment", ""),
                        
                    )
                )
            else:
                self.logger.warning(f"Unknown response type: {item.get('type')}")
            result.append(response_line)
        return result

    def _init_static_variables_sync(self, available_tone: List[str], available_expression: List[str]) -> None:
        """获取在prompt中不变的变量： persona, response_requirements, response_format (同步版本)"""
        static_variables_file = self.config.get("static_variables_file", None)
        if not static_variables_file:
            raise ValueError("static_variables_file must be provided in main_chat config")
        with open(static_variables_file, "r", encoding="utf-8") as f:
            static_vars: Dict[str, Any] = json.load(f)

        self.persona = static_vars.get("persona")
        if isinstance(self.persona, list):
            self.persona = "\n".join(self.persona)
        self.response_requirements = static_vars.get("response_requirements")
        if isinstance(self.response_requirements, list):
            self.response_requirements = "\n".join(self.response_requirements)

        response_format_raw = static_vars.get("response_format")
        if isinstance(response_format_raw, list):
            response_format_raw = "\n".join(response_format_raw)

        template = Template(response_format_raw)
        self.response_format = template.render(available_tone=available_tone, available_expression=available_expression)
