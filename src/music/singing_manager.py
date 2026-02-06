from ..utils.logger import get_logger
import pathlib
import os
import json
import io
import base64
import traceback
from ..types.music_type import SongSegment, SongMetadata, OneLyricLine
from ..types.tool_type import SiliconFlowTool, SiliconFlowFunction, SiliconFlowParameters, SiliconFlowOneParameter, MyTool
from typing import List, Tuple, Dict


class SingingManager:
    def __init__(self, config):
        self.logger = get_logger(__name__)
        self.config = config
        self.resource_path = config.get("resource_path", "res/music")
        self.all_songs: dict[str, SongMetadata] = {}
        self.tools: Dict[str, MyTool] = {}
        self.get_music_data()
        self.init_llm_tools()

    def get_music_data(self):
        self.logger.info(f"Loading music data from {self.resource_path}")
        music_lib = pathlib.Path(self.resource_path) / "songs"
        if not music_lib.exists():
            self.logger.warning(f"Music library path does not exist: {music_lib}")
            return

        for song in os.listdir(music_lib):
            song_dir = music_lib / song
            if not song_dir.is_dir():  # 安全名字即歌夹名称
                continue
            # 一首歌的文件包括：歌词文件 .lrc，音频文件 .mp3 或 .wav 以及配置文件 .json
            lyrics_file = song_dir / f"{song}.lrc"
            audio_file_wav = song_dir / f"{song}.mp3"
            config_file = song_dir / f"{song}.json"
            if not lyrics_file.exists():
                self.logger.warning(f"Lyrics file missing for song {song}")
                continue
            if not audio_file_wav.exists():
                self.logger.warning(f"Audio file missing for song {song}")
                continue
            if not config_file.exists():
                self.logger.warning(f"Config file missing for song {song}")
                continue

            # 读取配置文件
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    song_config = json.load(f)
                title = song_config.get("title", song)
                description = song_config.get("description", "")
                lrc_offset = song_config.get("lrc_offset", 0)
                segments = song_config.get("segments", [])
                segment_objs = []
                for seg in segments:
                    segment_objs.append(
                        SongSegment(
                            description=seg.get("description", ""),
                            start_time=seg.get("start_time", 0),
                            end_time=seg.get("end_time", 0),
                            lyrics=seg.get("lyrics", ""),
                        )
                    )
                song_metadata = SongMetadata(
                    title=title,
                    description=description,
                    song_path=str(audio_file_wav),
                    lrc_path=str(lyrics_file),
                    lrc_offset=lrc_offset,
                    segments=segment_objs,
                )
                self.all_songs[song] = song_metadata

            except Exception as e:
                import traceback

                self.logger.error(f"Failed to load song {song} config: {e}\n{traceback.format_exc()}")
        self.logger.info(f"Loaded {len(self.all_songs)} songs into music manager.")

    def get_song_metadata(self, song_name: str) -> SongMetadata | None:
        if not song_name:
            return None
        return self.all_songs.get(song_name, None)

    def can_i_sing_song(self, song_name: str) -> str:
        """
        检查是否可以演唱指定歌曲，如果可以，返回能够唱的唱段列表，否则返回空列表
        """
        if not song_name:
            return "没有指定歌曲名称。"
        safe_song_name = song_name.strip("").strip("《》")
        if not safe_song_name in self.all_songs:
            return "洛天依暂时不会唱这首歌。"
        song_metadata: SongMetadata = self.all_songs[safe_song_name]
        return f"{song_name}可以唱的选段：" + json.dumps(
            [segment.description for segment in song_metadata.segments], ensure_ascii=False
        )

    def get_all_song(self) -> str:
        song_and_desc = {}
        for song_name, metadata in self.all_songs.items():
            song_and_desc[song_name] = metadata.description

        # to json string
        return json.dumps(song_and_desc, ensure_ascii=False)

    def get_song_segment(self, song_name: str, segment_description: str) -> Tuple[List[OneLyricLine], bytes]:
        """
        根据歌曲名称和唱段描述，获取对应唱段的歌词对象列表，并返回音频数据的base64编码
        """
        if not song_name or not segment_description:
            return None, None

        # 移除书名号等干扰字符
        safe_song_name = song_name.strip().strip("《》")
        song_metadata = self.get_song_metadata(safe_song_name)

        if not song_metadata:
            self.logger.warning(f"Song not found: {song_name}")
            return None, None

        target_segment = None
        for seg in song_metadata.segments:
            if seg.description == segment_description:
                target_segment = seg
                break

        if not target_segment:
            self.logger.warning(f"Segment '{segment_description}' not found in song '{song_name}'")
            return None, None

        # 转换 lyrics (如果是 dict 则转换为 OneLyricLine)
        real_lyrics = []
        if target_segment.lyrics:
            first_elem = target_segment.lyrics[0]
            if isinstance(first_elem, dict):
                for l in target_segment.lyrics:
                    real_lyrics.append(OneLyricLine(duration=float(l.get("duration", 0.0)), content=str(l.get("content", ""))))
            elif isinstance(first_elem, OneLyricLine):
                real_lyrics = target_segment.lyrics

        # 处理音频
        try:
            from pydub import AudioSegment
        except ImportError:
            self.logger.error("pydub module not found. Please install it using 'pip install pydub'.")
            return None, None

        audio_path = song_metadata.song_path
        if not os.path.exists(audio_path):
            self.logger.error(f"Audio file does not exist: {audio_path}")
            return None, None

        try:
            # 加载并切片
            audio: AudioSegment = AudioSegment.from_file(audio_path)

            # 时间单位转换 (秒 -> 毫秒)
            start_ms = int(target_segment.start_time * 1000)
            end_ms = int(target_segment.end_time * 1000)

            segment_audio = audio[start_ms:end_ms]

            # 调整音量至目标 dBFS
            target_dbfs = -26.48
            change_in_dbfs = target_dbfs - segment_audio.dBFS
            segment_audio = segment_audio.apply_gain(change_in_dbfs)

            # 导出为 WAV 格式的 bytes
            wav_io = io.BytesIO()
            segment_audio.export(wav_io, format="wav")
            wav_bytes = wav_io.getvalue()

            # Base64 编码
            b64_encoded = base64.b64encode(wav_bytes)

            return real_lyrics, b64_encoded

        except Exception as e:
            self.logger.error(f"Failed to process audio for {song_name}: {e}\n{traceback.format_exc()}")
            return None, None

    def init_llm_tools(self) -> None:
        get_all_songs_tool = MyTool(
            name="get_all_songs",
            description="获取洛天依可以演唱的所有歌曲名称和描述。",
            tool_func=self.get_all_song,
            tool_interface=SiliconFlowTool(
                type="function",
                function=SiliconFlowFunction(
                    name="get_all_songs",
                    description="获取洛天依可以演唱的所有歌曲名称和简介",
                    parameters=SiliconFlowParameters(type="object", properties={}, required=[]),
                ),
            ),
        )

        can_i_sing_tool = MyTool(
            name="can_i_sing_song",
            description="检查洛天依是否可以演唱指定的歌曲，如果可以，返回能够唱的唱段列表，否则返回空列表。",
            tool_func=self.can_i_sing_song,
            tool_interface=SiliconFlowTool(
                type="function",
                function=SiliconFlowFunction(
                    name="can_i_sing_song",
                    description="检查洛天依是否可以演唱指定的歌曲，如果可以，返回能够唱的唱段列表，否则返回空列表。",
                    parameters=SiliconFlowParameters(
                        type="object",
                        properties={
                            "song_name": SiliconFlowOneParameter(type="string", description="歌曲名称"),
                        },
                        required=["song_name"],
                    ),
                ),
            ),
        )
        self.tools[get_all_songs_tool.name] = get_all_songs_tool
        self.tools[can_i_sing_tool.name] = can_i_sing_tool

    def get_tool_names(self) -> List[str]:
        return list(self.tools.keys())
    
    def get_tools(self) -> Dict[str, MyTool]:
        return self.tools