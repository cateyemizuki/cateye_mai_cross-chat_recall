"""cateye_mai_cross-chat_recall —— 跨聊私聊回忆。

让 bot 在必要情况下感知与指定群友的私聊记录：
- 扫描：按配置时机（planner 触发 / 消息推进 / 触发回复）对群内活跃用户与 bot 的私聊做一轮扫描；
- 总结：私聊满足条件（窗口内用户消息数达标）时交给 LLM 总结并写入插件自建 SQLite；
- 注入：把私聊概括（起止时间 + 对象 + 概括正文）注入群聊 planner / replyer 的
  上下文（首条 SystemMessageItem 之后），planner 与 reply 可分别开关；
- 工具：LLM 工具 recall_private_chat 按昵称主动召回私聊概括（候选只取当前会话上下文），
  召回成功后按配置的粘性时长继续随 reply 注入。

注意区分群聊与私聊：扫描与注入只发生在群聊会话；私聊内容只读不改。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple

from maibot_sdk import Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import HookMode, HookOrder, ToolParameterInfo, ToolParamType

from .helpers import allowed_by_list, extract_message_fields, msg_timestamp, unwrap_messages
from .injector import build_injection_block, build_recall_content, inject_after_first_system
from .logging_setup import get_logger, setup_file_logging
from .name_resolver import collect_candidates, resolve_user_name
from .scanner import Scanner
from .store import SummaryStore
from .summarizer import Summarizer

SUPPORTED_CONFIG_VERSION = "0.3.0"

_PLANNER_HOOK = "maisaka.planner.before_request"
_REPLYER_HOOK = "maisaka.replyer.before_model_request"
_RECEIVE_HOOK = "chat.receive.after_process"

_STREAM_CACHE_TTL = 60.0
_FOLLOW_PLANNER_TTL = 300.0
_FORCE_INJECT_TTL = 600.0  # /ccr debug test 强制注入标记的有效期（秒）


def _ui(en_label: str, en_hint: str = "") -> dict:
    """字段级英文翻译（并入 json_schema_extra）。"""
    entry: Dict[str, str] = {"label": en_label}
    if en_hint:
        entry["hint"] = en_hint
    return {"i18n": {"en": entry}}


# ---------------------------------------------------------------------------
# 配置模型
# ---------------------------------------------------------------------------

class PluginSectionConfig(PluginConfigBase):
    __ui_label__ = "插件"
    __ui_i18n__: ClassVar[Dict[str, Dict[str, str]]] = {
        "en": {"title": "Plugin", "description": "Base plugin settings."}
    }

    enabled: bool = Field(
        default=True,
        description="是否启用插件",
        json_schema_extra={
            "label": "启用插件",
            "hint": "关闭后停止所有扫描与注入，命令也不再响应",
            **_ui("Enabled", "Disable to stop all scanning and injection."),
        },
    )
    config_version: str = Field(
        default=SUPPORTED_CONFIG_VERSION,
        description="配置版本（与插件版本同步）",
        json_schema_extra={"hidden": True, "disabled": True},
    )


class ScanSectionConfig(PluginConfigBase):
    __ui_label__ = "扫描"
    __ui_i18n__: ClassVar[Dict[str, Dict[str, str]]] = {
        "en": {"title": "Scan", "description": "When and how broadly to scan members' private chats."}
    }

    trigger_mode: Literal["planner", "push", "reply"] = Field(
        default="planner",
        description="扫描触发方式（三选一）：planner=planner 触发时扫描；push=群聊新消息推进计数达标时扫描；reply=触发回复时只扫描被回复的目标用户",
        json_schema_extra={
            "label": "扫描触发方式",
            "hint": "planner=每次 planner 请求前扫一轮；push=群聊每新进 N 条非 bot 消息扫一轮；reply=只在回复某条消息时扫被回复的人",
            **_ui("Scan trigger", "planner / push / reply"),
        },
    )
    scan_count: int = Field(
        default=10,
        ge=1,
        le=100,
        description="扫描数：取群聊最近 N 条非 bot 消息汇总其中的用户 id 作为候选（push 模式下同时是推进计数阈值）；消息不足 N 条时回退为全部可用消息",
        json_schema_extra={
            "label": "扫描数",
            "hint": "决定扫哪些用户：最近 N 条非 bot 群消息里出现过的用户；push 模式下也是触发扫描的消息条数",
            **_ui("Scan count", "Recent non-bot messages per scan; push threshold"),
        },
    )


class SummarizeSectionConfig(PluginConfigBase):
    __ui_label__ = "总结"
    __ui_i18n__: ClassVar[Dict[str, Dict[str, str]]] = {
        "en": {"title": "Summarize", "description": "LLM summarization of private chats."}
    }

    min_user_messages: int = Field(
        default=5,
        ge=1,
        description="私聊触发总结的条件：总结时间窗口内来自用户的消息数达到该值（不含 bot 消息）",
        json_schema_extra={
            "label": "用户消息数阈值",
            "hint": "24 小时窗口内对方发的消息达到这么多条才会总结",
            **_ui("Min user messages", "User messages required in window"),
        },
    )
    window_hours: float = Field(
        default=24.0,
        ge=0.1,
        description="总结时间窗口（小时）：超出窗口的私聊消息一律不触发总结，也不会进入 LLM 总结请求体",
        json_schema_extra={
            "label": "时间窗口（小时）",
            "hint": "只总结最近这段时间的私聊；更早的消息不参与总结",
            **_ui("Window (hours)", "Only summarize chats within this window"),
        },
    )
    resummarize_push_count: int = Field(
        default=5,
        ge=1,
        description="允许被重新总结的判断：距上次总结新推进（不含 bot 消息）多少条用户消息后进入重总结队列；进入队列后不立刻总结，再次被扫描到才重新总结（全量重写覆盖）",
        json_schema_extra={
            "label": "重总结推进数",
            "hint": "对方再发这么多条消息后标记待重总结，等下次扫描时才真正重新总结",
            **_ui("Re-summarize push count", "New messages before re-summarize queue"),
        },
    )
    model: str = Field(
        default="replyer",
        description="总结使用的模型：填模型任务名（replyer/planner/utils 等，默认 replyer）或 WebUI 模型列表里的具体模型名；留空走 utils 任务",
        json_schema_extra={
            "label": "总结模型",
            "hint": "填 replyer/planner 等任务名走对应任务路由；或填具体模型名（失去自动切换能力）；留空=utils",
            "placeholder": "replyer",
            **_ui("Summary model", "Task name (replyer) or concrete model name; empty = utils"),
        },
    )
    max_tokens: int = Field(
        default=1024,
        ge=0,
        description="总结 LLM 请求的最大 token 数（0 = 不限制）",
        json_schema_extra={
            "label": "最大 token 数",
            "hint": "0 表示不限制；默认提示词约输出 400 字，1024 足够",
            **_ui("Max tokens", "0 = unlimited"),
        },
    )
    prompt: str = Field(
        default="",
        description="总结提示词（system 角色，发给 LLM 的总结要求）；留空使用插件内置默认提示词。支持占位符：{bot_name}=bot 名字，{user_nickname}=对方昵称，{user_id}=对方 QQ 号",
        json_schema_extra={
            "label": "总结提示词",
            "hint": "留空=内置默认（按时间线列关键事件+时间点+需记住的信息）；可用占位符 {bot_name} {user_nickname} {user_id}",
            "x-widget": "textarea",
            "rows": 8,
            "placeholder": "留空使用内置默认提示词",
            **_ui("Summary prompt", "Empty = built-in default; supports {bot_name}/{user_nickname}/{user_id}"),
        },
    )


class InjectSectionConfig(PluginConfigBase):
    __ui_label__ = "注入"
    __ui_i18n__: ClassVar[Dict[str, Dict[str, str]]] = {
        "en": {"title": "Inject", "description": "Which chats get the memory block, and for whom."}
    }

    inject_planner: bool = Field(
        default=True,
        description="是否把私聊概括注入 planner 上下文（首条 SystemMessageItem 之后）",
        json_schema_extra={
            "label": "注入 planner",
            "hint": "决定 bot 的决策模型能否想起私聊内容",
            **_ui("Inject into planner"),
        },
    )
    inject_reply: bool = Field(
        default=True,
        description="是否把私聊概括注入 reply 上下文（首条 SystemMessageItem 之后）",
        json_schema_extra={
            "label": "注入 reply",
            "hint": "决定 bot 回复时能否想起私聊内容",
            **_ui("Inject into reply"),
        },
    )
    planner_user_mode: Literal["trigger", "recent_users"] = Field(
        default="trigger",
        description="planner 注入的用户选择：trigger=按触发 planner 的消息的归属用户；recent_users=按触发 planner 的消息最近的 N 条非 bot 消息的用户（N = 下方的 planner 最近用户数）",
        json_schema_extra={
            "label": "planner 注入用户",
            "hint": "trigger=只看触发消息是谁发的；recent_users=看最近 N 条群消息里都有谁",
            **_ui("Planner user mode", "trigger = triggering user; recent_users = users of recent N messages"),
        },
    )
    recent_user_count: int = Field(
        default=5,
        ge=1,
        le=50,
        description="planner 注入用户选择为 recent_users 模式时，取触发 planner 的消息最近的 N 条非 bot 消息中包含的用户（默认 5，可修改）",
        json_schema_extra={
            "label": "planner 最近用户数",
            "hint": "recent_users 模式下取最近多少条非 bot 群消息里出现过的用户（默认 5）",
            **_ui("Planner recent users", "Users from recent N non-bot messages"),
        },
    )
    reply_user_mode: Literal["reply_source", "follow_planner"] = Field(
        default="reply_source",
        description="reply 注入的用户选择：reply_source=按要回复的消息来源用户；follow_planner=跟随 planner 注入的用户",
        json_schema_extra={
            "label": "reply 注入用户",
            "hint": "reply_source=回复谁就注入与谁的私聊；follow_planner=与 planner 注入同一批人",
            **_ui("Reply user mode", "reply_source / follow_planner"),
        },
    )


class StoreSectionConfig(PluginConfigBase):
    __ui_label__ = "总结库"
    __ui_i18n__: ClassVar[Dict[str, Dict[str, str]]] = {
        "en": {"title": "Summary Store", "description": "Retention of stored summaries."}
    }

    summary_ttl_hours: float = Field(
        default=24.0,
        ge=0.1,
        description="总结内容截止时间超出多久后丢弃（小时）；过期总结会在扫描时删除且不再注入",
        json_schema_extra={
            "label": "总结保留时长（小时）",
            "hint": "总结里最后一次聊天时间距今超过这个时长就作废删除",
            **_ui("Summary TTL (hours)", "Discard summaries older than this"),
        },
    )


class ScopeSectionConfig(PluginConfigBase):
    __ui_label__ = "作用范围"
    __ui_i18n__: ClassVar[Dict[str, Dict[str, str]]] = {
        "en": {"title": "Scope", "description": "Whitelist / blacklist. Empty whitelist = allow all; blacklist wins."}
    }

    group_whitelist: List[str] = Field(
        default_factory=list,
        description="群白名单（群号列表）：非空时仅对这些群生效",
        json_schema_extra={
            "label": "群白名单",
            "hint": "留空=所有群生效；填写群号则只在名单内的群工作",
            **_ui("Group whitelist", "Empty = all groups"),
        },
    )
    group_blacklist: List[str] = Field(
        default_factory=list,
        description="群黑名单（群号列表）：命中则不生效（优先级高于白名单）",
        json_schema_extra={
            "label": "群黑名单",
            "hint": "名单内的群禁用本插件；优先级高于白名单",
            **_ui("Group blacklist", "Blacklisted groups are excluded"),
        },
    )
    user_whitelist: List[str] = Field(
        default_factory=list,
        description="用户白名单（QQ 号列表）：非空时仅对这些用户扫描/注入",
        json_schema_extra={
            "label": "用户白名单",
            "hint": "留空=所有群友；填写 QQ 号则只对这些人做私聊回忆",
            **_ui("User whitelist", "Empty = all users"),
        },
    )
    user_blacklist: List[str] = Field(
        default_factory=list,
        description="用户黑名单（QQ 号列表）：命中则不扫描/注入（优先级高于白名单）",
        json_schema_extra={
            "label": "用户黑名单",
            "hint": "名单内的用户禁用私聊回忆；优先级高于白名单",
            **_ui("User blacklist", "Blacklisted users are excluded"),
        },
    )


class ToolSectionConfig(PluginConfigBase):
    __ui_label__ = "主动召回"
    __ui_i18n__: ClassVar[Dict[str, Dict[str, str]]] = {
        "en": {"title": "Proactive Recall", "description": "On-demand recall tool and its sticky retention."}
    }

    recall_retention_minutes: float = Field(
        default=5.0,
        ge=0.5,
        le=1440,
        description="粘性召回保留分钟数：工具成功召回后，该用户的私聊概括会在这段时间内继续注入 reply 上下文；再次成功召回会刷新",
        json_schema_extra={
            "label": "粘性召回保留（分钟）",
            "hint": "工具召回成功后，这段时间内的回复会继续带上该用户的私聊概括；期间再次召回会重新计时",
            **_ui("Sticky recall (minutes)", "How long a successful recall keeps injecting"),
        },
    )
    context_window_messages: int = Field(
        default=50,
        ge=5,
        le=200,
        description="名称候选取最近多少条群消息：工具按这个范围解析用户昵称，只认当前会话上下文里出现过的用户",
        json_schema_extra={
            "label": "候选上下文消息数",
            "hint": "只看当前聊流最近这么多条消息里出现过的昵称；调大=能认出更早发言的人，调小=更省",
            **_ui("Context window messages", "Recent messages used to resolve user names"),
        },
    )


class CrossChatRecallConfig(PluginConfigBase):
    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    scan: ScanSectionConfig = Field(default_factory=ScanSectionConfig)
    summarize: SummarizeSectionConfig = Field(default_factory=SummarizeSectionConfig)
    inject: InjectSectionConfig = Field(default_factory=InjectSectionConfig)
    store: StoreSectionConfig = Field(default_factory=StoreSectionConfig)
    scope: ScopeSectionConfig = Field(default_factory=ScopeSectionConfig)
    tool: ToolSectionConfig = Field(default_factory=ToolSectionConfig)


# ---------------------------------------------------------------------------
# 插件主体
# ---------------------------------------------------------------------------

class CrossChatRecallPlugin(MaiBotPlugin):
    config_model: ClassVar[type[PluginConfigBase] | None] = CrossChatRecallConfig

    def __init__(self) -> None:
        super().__init__()
        self._file_log = get_logger("plugin")
        self._store: Optional[SummaryStore] = None
        self._scanner: Optional[Scanner] = None
        self._summarizer: Optional[Summarizer] = None
        self._bg_tasks: set = set()
        self._scan_lock: Optional[asyncio.Lock] = None
        self._stream_cache: Dict[str, Mapping[str, Any]] = {}
        self._stream_cache_ts: float = 0.0
        self._planner_users_cache: Dict[str, Tuple[float, List[str]]] = {}
        self._push_counters: Dict[str, int] = {}
        # /ccr debug test 强制注入标记：session_id -> {"planner": (ts, uid), "reply": (ts, uid)}
        self._force_inject: Dict[str, Dict[str, Tuple[float, str]]] = {}
        # 工具主动召回的粘性标记：stream_id -> {user_id: 过期时间戳}（内存态，不持久化）
        self._tool_recall_sticky: Dict[str, Dict[str, float]] = {}
        # 扫描被跳过（锁忙）的会话，等当前扫描结束后补触发
        self._pending_scan: set = set()
        self._bot_name: str = ""

    # -- 生命周期 -----------------------------------------------------------

    async def on_load(self) -> None:
        # 调试文件日志（插件目录 logs/，不符合插件规范，测试期专用）
        try:
            plugin_dir = Path(__file__).resolve().parent
            setup_file_logging(plugin_dir)
        except Exception:
            pass

        self._scan_lock = asyncio.Lock()
        self._store = SummaryStore(self.ctx.paths.data_dir / "cross_chat_recall.db")
        self._scanner = Scanner(self)
        self._summarizer = Summarizer(self)

        await self._load_bot_name()
        cfg = self.config
        self._file_log.info(
            "插件加载完成 v%s trigger_mode=%s scan_count=%s inject(planner=%s, reply=%s, "
            "planner_mode=%s, recent_users=%s, reply_mode=%s) min_user_msgs=%s window=%sh "
            "resummarize_push=%s model=%s ttl=%sh tool(recall_sticky=%smin, ctx_msgs=%s)",
            SUPPORTED_CONFIG_VERSION,
            cfg.scan.trigger_mode, cfg.scan.scan_count,
            cfg.inject.inject_planner, cfg.inject.inject_reply,
            cfg.inject.planner_user_mode, cfg.inject.recent_user_count, cfg.inject.reply_user_mode,
            cfg.summarize.min_user_messages, cfg.summarize.window_hours,
            cfg.summarize.resummarize_push_count, cfg.summarize.model, cfg.store.summary_ttl_hours,
            cfg.tool.recall_retention_minutes, cfg.tool.context_window_messages,
        )
        self._file_log.info("bot 名称=%s", self._bot_name)

    async def on_unload(self) -> None:
        tasks = [t for t in self._bg_tasks if not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            # 等被取消的任务收尾，再关库，避免残余写库撞已关闭连接
            await asyncio.gather(*tasks, return_exceptions=True)
        self._bg_tasks.clear()
        if self._store is not None:
            self._store.close()
            self._store = None
        self._file_log.info("插件已卸载，后台任务已取消")

    async def on_config_update(self, scope: str, config_data: dict, version: str) -> None:
        self._file_log.info("配置热更新 scope=%s version=%s", scope, version)
        if scope == "self" and self.config:
            self._file_log.info(
                "新配置 trigger_mode=%s inject(planner=%s, reply=%s) scan_count=%s",
                self.config.scan.trigger_mode,
                self.config.inject.inject_planner,
                self.config.inject.inject_reply,
                self.config.scan.scan_count,
            )

    # -- 对内属性 -----------------------------------------------------------

    @property
    def store(self) -> SummaryStore:
        assert self._store is not None, "插件未完成加载"
        return self._store

    @property
    def scanner(self) -> Scanner:
        assert self._scanner is not None, "插件未完成加载"
        return self._scanner

    @property
    def summarizer(self) -> Summarizer:
        assert self._summarizer is not None, "插件未完成加载"
        return self._summarizer

    @property
    def bot_name(self) -> str:
        return self._bot_name or "bot"

    async def _load_bot_name(self) -> None:
        for key in ("bot.nickname", "person.bot_name", "person.nickname"):
            try:
                value = await self.ctx.config.get(key)
            except Exception:
                value = None
            if value:
                self._bot_name = str(value).strip()
                self._file_log.info("读取 bot 名称 %s=%r", key, self._bot_name)
                return
        self._bot_name = "bot"
        self._file_log.warning("未能从宿主配置读取 bot 名称，使用兜底值 %r", self._bot_name)

    # -- 流信息缓存 ---------------------------------------------------------

    async def get_stream(self, session_id: str) -> Optional[Mapping[str, Any]]:
        """按 session_id（=stream_id）查聊天流，带 60s 缓存。"""
        if not session_id:
            return None
        now = time.time()
        if self._stream_cache and now - self._stream_cache_ts < _STREAM_CACHE_TTL:
            return self._stream_cache.get(session_id)
        try:
            streams = await self.ctx.chat.get_all_streams()
        except Exception as exc:
            self._file_log.warning("get_all_streams 异常: %s", exc)
            return self._stream_cache.get(session_id)
        mapping: Dict[str, Mapping[str, Any]] = {}
        for stream in streams or []:
            if not isinstance(stream, Mapping):
                continue
            sid = str(stream.get("stream_id") or stream.get("session_id") or "")
            if sid:
                mapping[sid] = stream
        self._stream_cache = mapping
        self._stream_cache_ts = now
        self._file_log.debug("聊天流缓存已刷新：%d 条", len(mapping))
        return mapping.get(session_id)

    async def _is_group_session(self, session_id: str) -> Tuple[bool, Optional[Mapping[str, Any]]]:
        stream = await self.get_stream(session_id)
        if not stream:
            return False, None
        is_group = bool(stream.get("is_group_session")) or str(stream.get("chat_type") or "") == "group"
        return is_group, stream

    def _group_allowed(self, group_id: str) -> bool:
        scope = self.config.scope
        return allowed_by_list(group_id, scope.group_whitelist, scope.group_blacklist)

    def _user_allowed(self, user_id: str) -> bool:
        scope = self.config.scope
        return allowed_by_list(user_id, scope.user_whitelist, scope.user_blacklist)

    # -- 近期群消息 ---------------------------------------------------------

    async def _recent_group_messages(self, session_id: str, want: int) -> List[Mapping[str, Any]]:
        """取群聊最近 want 条非 bot、非合成的消息（新到旧）。"""
        try:
            raw = await self.ctx.message.get_recent(chat_id=session_id, limit=max(want * 3, 30))
        except Exception as exc:
            self._file_log.warning("群 %s：get_recent 异常: %s", session_id, exc)
            return []
        messages = unwrap_messages(raw, self._file_log, "get_recent")
        fresh: List[Mapping[str, Any]] = []
        for msg in messages:
            fields = extract_message_fields(msg)
            if fields.get("is_self") or fields.get("is_synthetic") or not fields.get("user_id"):
                continue
            fresh.append(msg)
        fresh.sort(key=msg_timestamp, reverse=True)
        if len(fresh) > want:
            fresh = fresh[:want]
        return fresh

    # -- 注入用户选择 -------------------------------------------------------

    async def _select_planner_users(self, session_id: str) -> List[str]:
        """按 planner_user_mode 选注入用户（已过用户黑白名单）。"""
        mode = self.config.inject.planner_user_mode
        if mode == "recent_users":
            want = int(self.config.inject.recent_user_count)
        else:
            want = 1
        recent = await self._recent_group_messages(session_id, want)
        if not recent:
            self._file_log.info("planner 注入：群 %s 最近无非 bot 消息，无法选用户", session_id)
            return []
        users: List[str] = []
        for msg in recent:
            uid = extract_message_fields(msg).get("user_id") or ""
            if uid and uid not in users:
                users.append(uid)
            if len(users) >= want:
                break
        trigger_user = users[0] if users else ""
        users = [u for u in users if self._user_allowed(u)]
        self._file_log.info(
            "planner 注入：群 %s mode=%s 触发用户=%s 选中=%s", session_id, mode, trigger_user, users,
        )
        return users

    async def _lookup_reply_user(self, session_id: str, reply_message_id: Any) -> Optional[str]:
        """通过被回复的消息 id 反查用户 id。"""
        mid = str(reply_message_id or "").strip()
        if not mid:
            return None
        try:
            msg = await self.ctx.message.get_by_id(mid, chat_id=session_id, stream_id=session_id)
        except Exception as exc:
            self._file_log.warning("reply 注入：反查消息 %s 异常: %s", mid, exc)
            return None
        if not isinstance(msg, Mapping):
            self._file_log.info("reply 注入：被回复消息 %s 不存在或形态异常", mid)
            return None
        uid = extract_message_fields(msg).get("user_id") or ""
        if not uid:
            return None
        self._file_log.info("reply 注入：被回复消息 %s → 用户 %s", mid, uid)
        return uid

    # -- 注入执行 -----------------------------------------------------------

    def _collect_valid_summaries(self, users: List[str]) -> List[Dict[str, Any]]:
        """取这些用户当前有效（未过期、非空）的总结，按最近私聊时间新→旧排序。"""
        ttl = float(self.config.store.summary_ttl_hours) * 3600
        cutoff = time.time() - ttl
        rows = self.store.get_many(users)
        entries: List[Dict[str, Any]] = []
        for uid in users:
            row = rows.get(uid)
            if not row:
                continue
            if float(row.get("end_ts") or 0) < cutoff:
                self._file_log.debug("注入：用户 %s 的总结已过 TTL，跳过", uid)
                continue
            summary = str(row.get("summary") or "").strip()
            if not summary:
                continue
            entries.append({
                "user_id": uid,
                "nickname": row.get("user_nickname") or uid,
                "start_ts": row.get("start_ts") or 0,
                "end_ts": row.get("end_ts") or 0,
                "summary": summary,
            })
        entries.sort(key=lambda e: float(e.get("end_ts") or 0), reverse=True)
        return entries

    def _apply_injection(self, kwargs: Dict[str, Any], users: List[str], scene: str) -> Dict[str, Any]:
        """把 users 的总结合并为一个块插入 items；返回待返回的 hook 结果（无内容则仅 continue）。"""
        entries = self._collect_valid_summaries(users)
        if not entries:
            self._file_log.info("%s 注入：选中的用户 %s 均无有效总结，跳过", scene, users)
            return {"action": "continue"}
        block = build_injection_block(entries)
        items = inject_after_first_system(kwargs.get("items"), block, self._file_log)
        if items is None:
            return {"action": "continue"}
        modified = dict(kwargs)
        modified["items"] = items
        self._file_log.info(
            "%s 注入：已插入私聊记忆块（%d 个用户：%s，块 %d 字符）",
            scene, len(entries), [e["user_id"] for e in entries], len(block),
        )
        self._file_log.debug("%s 注入块全文：\n%s", scene, block)
        return {"action": "continue", "modified_kwargs": modified}

    # -- 工具召回粘性保留 ---------------------------------------------------

    def _register_sticky_recall(self, stream_id: str, user_id: str) -> None:
        """注册/刷新粘性召回：该用户的私聊概括在保留期内继续随 reply 注入（内存态，不落盘）。"""
        now = time.time()
        minutes = float(self.config.tool.recall_retention_minutes)
        entry = self._tool_recall_sticky.setdefault(stream_id, {})
        entry[user_id] = now + minutes * 60
        self._prune_sticky_recall(stream_id, now)
        self._file_log.info(
            "工具召回：注册粘性召回 stream=%s user=%s 保留 %.1f 分钟", stream_id, user_id, minutes,
        )

    def _prune_sticky_recall(self, stream_id: str, now: Optional[float] = None) -> None:
        """惰性清理某聊流的过期粘性条目（全部过期时移除整条记录）。"""
        entry = self._tool_recall_sticky.get(stream_id)
        if not entry:
            return
        current = now if now is not None else time.time()
        for uid in [uid for uid, expire_at in entry.items() if expire_at <= current]:
            entry.pop(uid, None)
        if not entry:
            self._tool_recall_sticky.pop(stream_id, None)

    def _sticky_users(self, stream_id: str) -> List[str]:
        """取该聊流当前未过期的粘性召回用户（顺序 = 注册先后）。"""
        self._prune_sticky_recall(stream_id)
        return list((self._tool_recall_sticky.get(stream_id) or {}).keys())

    # -- 强制注入（/ccr debug test）-----------------------------------------

    def _set_force_inject(self, session_id: str, user_id: str) -> None:
        """为当前聊流的下一次 planner 与 reply 请求设置强制注入标记。"""
        now = time.time()
        entry = self._force_inject.setdefault(session_id, {})
        entry["planner"] = (now, user_id)
        entry["reply"] = (now, user_id)
        self._file_log.info("强制注入标记已设置 session=%s user=%s", session_id, user_id)

    def _peek_force_inject(self, session_id: str, scene: str) -> Optional[str]:
        """读取强制注入标记（不消费）；过期自动清除。返回 None 表示无标记。"""
        entry = self._force_inject.get(session_id)
        if not entry:
            return None
        item = entry.get(scene)
        if not item:
            return None
        ts, uid = item
        if time.time() - ts > _FORCE_INJECT_TTL:
            self._file_log.info("强制注入标记已过期 session=%s scene=%s", session_id, scene)
            entry.pop(scene, None)
            if not entry:
                self._force_inject.pop(session_id, None)
            return None
        return uid

    def _consume_force_inject(self, session_id: str, scene: str) -> None:
        """注入成功后消费标记。"""
        entry = self._force_inject.get(session_id)
        if entry:
            entry.pop(scene, None)
            if not entry:
                self._force_inject.pop(session_id, None)
        self._file_log.info("强制注入已生效并消费标记 session=%s scene=%s", session_id, scene)

    # -- 扫描调度 -----------------------------------------------------------

    def _schedule_scan(self, session_id: str, *, focus_user_ids: Optional[List[str]] = None, reason: str = "") -> None:
        task = asyncio.create_task(self._scan_runner(session_id, focus_user_ids, reason))
        self._bg_tasks.add(task)
        task.add_done_callback(self._on_bg_done)

    def _on_bg_done(self, task: "asyncio.Task[Any]") -> None:
        self._bg_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self._file_log.error("后台扫描任务异常: %s", task.exception())

    async def _scan_runner(self, session_id: str, focus_user_ids: Optional[List[str]], reason: str) -> None:
        assert self._scan_lock is not None
        if self._scan_lock.locked():
            # 已有扫描在进行：登记补触发，避免该批消息漏扫（计数已清零）
            self._pending_scan.add(session_id)
            self._file_log.info(
                "[扫描] 已有扫描在进行，登记补触发 stream=%s reason=%r", session_id, reason,
            )
            return
        try:
            async with self._scan_lock:
                await self.scanner.scan_stream(session_id, focus_user_ids=focus_user_ids, reason=reason)
        finally:
            if session_id in self._pending_scan:
                self._pending_scan.discard(session_id)
                self._file_log.info("[扫描] 锁已释放，补触发 stream=%s", session_id)
                self._schedule_scan(session_id, focus_user_ids=focus_user_ids, reason="补触发")

    # -- Hook：planner ------------------------------------------------------

    @HookHandler(_PLANNER_HOOK, name="ccr_planner_inject", mode=HookMode.BLOCKING, order=HookOrder.NORMAL)
    async def on_planner_before_request(self, **kwargs: Any) -> Dict[str, Any]:
        if not self.config.plugin.enabled:
            return {"action": "continue"}
        session_id = str(kwargs.get("session_id") or "")
        if not session_id:
            return {"action": "continue"}

        # /ccr debug test 强制注入：绕过群聊/范围/开关检查（调试专用）
        force_user = self._peek_force_inject(session_id, "planner")

        is_group, stream = await self._is_group_session(session_id)
        if force_user is None:
            if not is_group or not stream:
                self._file_log.debug("planner：会话 %s 非群聊，跳过（区分群聊/私聊）", session_id)
                return {"action": "continue"}
            group_id = str(stream.get("group_id") or "")
            if not self._group_allowed(group_id):
                self._file_log.debug("planner：群 %s 不在作用范围，跳过", group_id)
                return {"action": "continue"}

        # 扫描模式 = planner 触发时：本轮顺带触发扫描（后台执行，不阻塞 planner）
        if self.config.scan.trigger_mode == "planner":
            self._file_log.info("planner 触发扫描：群 %s（session=%s）",
                                (stream or {}).get("group_id") or "?", session_id)
            self._schedule_scan(session_id, reason="planner 触发")

        # 强制注入优先（标记在注入成功后才消费；总结未就绪时保留并走常规路径）
        if force_user is not None:
            result = self._apply_injection(dict(kwargs), [force_user], "planner(强制)")
            if "modified_kwargs" in result:
                self._consume_force_inject(session_id, "planner")
                return result
            self._file_log.info("planner：强制注入暂无用户 %s 的有效总结，保留标记", force_user)

        if not self.config.inject.inject_planner:
            return {"action": "continue"}

        # 注入用户选择（结果同时缓存给 follow_planner 模式的 reply 使用）
        users = await self._select_planner_users(session_id)
        self._planner_users_cache[session_id] = (time.time(), list(users))
        if not users:
            return {"action": "continue"}
        return self._apply_injection(dict(kwargs), users, "planner")

    # -- Hook：replyer ------------------------------------------------------

    @HookHandler(_REPLYER_HOOK, name="ccr_replyer_inject", mode=HookMode.BLOCKING, order=HookOrder.NORMAL)
    async def on_replyer_before_model_request(self, **kwargs: Any) -> Dict[str, Any]:
        cfg = self.config
        if not cfg.plugin.enabled:
            return {"action": "continue"}
        session_id = str(kwargs.get("session_id") or "")
        if not session_id:
            return {"action": "continue"}

        # /ccr debug test 强制注入：绕过群聊/范围/开关检查（调试专用）
        force_user = self._peek_force_inject(session_id, "reply")

        # 工具主动召回注册的粘性用户（未过期；惰性清理）。注入阶段不再过群/用户黑白名单
        sticky_users = self._sticky_users(session_id)

        is_group, stream = await self._is_group_session(session_id)
        group_id = str((stream or {}).get("group_id") or "")
        scope_ok = bool(is_group and stream and self._group_allowed(group_id))
        if force_user is None and not scope_ok:
            # 常规路径因"非群聊/不在作用范围"被跳过：粘性用户在工具调用时已过候选校验，
            # 只要未过期且 inject_reply 开启，仍按粘性块注入
            if sticky_users and cfg.inject.inject_reply:
                self._file_log.info(
                    "reply 注入：常规路径跳过（非群聊或群 %s 不在作用范围），仅注入粘性召回用户 %s",
                    group_id or "?", sticky_users,
                )
                return self._apply_injection(dict(kwargs), list(sticky_users), "replyer(粘性)")
            self._file_log.debug("replyer：会话 %s 非群聊/不在作用范围，跳过（区分群聊/私聊）", session_id)
            return {"action": "continue"}

        # 反查被回复的目标用户（reply 扫描与 reply_source 注入共用一次 RPC，省 6000ms 超时预算）
        reply_user: Optional[str] = None
        if cfg.scan.trigger_mode == "reply" or cfg.inject.reply_user_mode == "reply_source":
            reply_user = await self._lookup_reply_user(session_id, kwargs.get("reply_message_id"))

        # 扫描模式 = 触发回复时：只扫描被回复的目标用户（reply_message_id 反查）
        if cfg.scan.trigger_mode == "reply":
            if reply_user:
                self._file_log.info("reply 触发定向扫描：群 %s 用户 %s",
                                    (stream or {}).get("group_id") or "?", reply_user)
                self._schedule_scan(session_id, focus_user_ids=[reply_user], reason="reply 触发")
            else:
                self._file_log.info("reply 触发扫描：本次回复无可反查的目标消息，跳过扫描")

        # 强制注入优先（标记在注入成功后才消费；总结未就绪时保留并走常规路径）
        if force_user is not None:
            result = self._apply_injection(dict(kwargs), [force_user], "replyer(强制)")
            if "modified_kwargs" in result:
                self._consume_force_inject(session_id, "reply")
                return result
            self._file_log.info("replyer：强制注入暂无用户 %s 的有效总结，保留标记", force_user)

        if not cfg.inject.inject_reply:
            return {"action": "continue"}

        # 注入用户选择
        users: List[str] = []
        if cfg.inject.reply_user_mode == "follow_planner":
            cached = self._planner_users_cache.get(session_id)
            if cached and time.time() - cached[0] < _FOLLOW_PLANNER_TTL:
                users = list(cached[1])
                self._file_log.info("reply 注入：跟随 planner 用户 %s（session=%s）", users, session_id)
            else:
                self._file_log.info("reply 注入：follow_planner 无有效缓存（session=%s），本次不注入", session_id)
        else:
            if reply_user and self._user_allowed(reply_user):
                users = [reply_user]
            elif reply_user:
                self._file_log.info("reply 注入：用户 %s 不在作用范围", reply_user)

        # 常规选中用户与粘性召回用户取并集（去重），统一注入；粘性用户不再过黑白名单
        if sticky_users:
            merged = list(dict.fromkeys([*users, *sticky_users]))
            self._file_log.info(
                "reply 注入：常规用户 %s + 粘性用户 %s → 合并 %s（session=%s）",
                users, sticky_users, merged, session_id,
            )
            users = merged
        if not users:
            return {"action": "continue"}
        return self._apply_injection(dict(kwargs), users, "replyer")

    # -- 组件：LLM 工具（主动召回）------------------------------------------

    @Tool(
        "recall_private_chat",
        brief_description="召回你与某位群友的私聊记忆",
        detailed_description=(
            "当用户让你回忆与某人的私聊内容、或你需要主动想起与某人私下聊过什么时调用。\n"
            "仅在用户明确要求回忆/查找私聊内容时调用，不要主动调用，也不要在普通闲聊里调用。\n"
            "限制：只能查询当前会话上下文里出现过的用户；传入的名称应取自上下文中的用户昵称，"
            "不要凭空编造，也不要传 QQ 号。\n"
            "参数：user_name（string，必填）——要召回私聊记忆的用户昵称，取当前会话上下文中的名称。\n"
            "返回：success 表示是否找到私聊记忆，content 为召回内容或失败原因"
            "（如名称有误、匹配到多个用户、没有私聊内容等）；失败时 content 会给出原因，可直接转述给用户。"
        ),
        parameters=[
            ToolParameterInfo(
                name="user_name",
                param_type=ToolParamType.STRING,
                description="要召回私聊记忆的用户昵称（取当前会话上下文中的名称）",
                required=True,
            ),
        ],
    )
    async def tool_recall_private_chat(self, user_name: Any = "", **kwargs: Any) -> Dict[str, Any]:
        stream_id = str(kwargs.get("stream_id") or "").strip()
        wanted = str(user_name or "").strip()

        # 1-3. 候选只取当前聊流上下文 → 精准匹配 → 正则降级（五分支见 name_resolver）
        candidates = await self._context_candidates(stream_id) if stream_id else []
        resolved = resolve_user_name(candidates, wanted)
        if resolved.get("status") == "ambiguous":
            self._file_log.info("工具 recall_private_chat(%r)：匹配到多个用户 %s", wanted, resolved.get("names"))
            return {"success": False, "content": str(resolved.get("message") or "")}
        if resolved.get("status") != "hit":
            self._file_log.info(
                "工具 recall_private_chat(%r)：名称有误（stream=%s 候选 %d 个）",
                wanted, stream_id or "-", len(candidates),
            )
            return {"success": False, "content": str(resolved.get("message") or "")}

        # 4. 命中唯一用户：查总结，无记录/为空/已过 TTL → 视为没有私聊，不注册粘性
        user_id = str(resolved.get("user_id") or "")
        matched_nickname = str(resolved.get("nickname") or "")
        row = self.store.get(user_id)
        summary = str((row or {}).get("summary") or "").strip()
        ttl_seconds = float(self.config.store.summary_ttl_hours) * 3600
        if not row or not summary or float(row.get("end_ts") or 0) < time.time() - ttl_seconds:
            self._file_log.info(
                "工具 recall_private_chat(%r)：用户 %s 无有效总结（无记录/为空/已过 TTL）", wanted, user_id,
            )
            return {"success": True, "content": "你和该用户没有私聊内容。"}

        # 5. 有有效总结：注册/刷新粘性召回，返回与注入块同风格的召回内容
        self._register_sticky_recall(stream_id, user_id)
        content = build_recall_content({
            "user_id": user_id,
            "nickname": str(row.get("user_nickname") or "").strip() or matched_nickname,
            "start_ts": row.get("start_ts") or 0,
            "end_ts": row.get("end_ts") or 0,
            "summary": summary,
        })
        self._file_log.info(
            "工具 recall_private_chat(%r)：用户 %s 召回成功（粘性保留 %.1f 分钟）",
            wanted, user_id, float(self.config.tool.recall_retention_minutes),
        )
        return {"success": True, "content": content}

    async def _context_candidates(self, stream_id: str) -> List[Tuple[str, str]]:
        """取当前聊流最近 tool.context_window_messages 条消息中出现过的用户昵称候选（新→旧）。"""
        limit = int(self.config.tool.context_window_messages)
        try:
            raw = await self.ctx.message.get_recent(chat_id=stream_id, limit=limit)
        except Exception as exc:
            self._file_log.warning("工具召回：stream=%s get_recent 异常: %s", stream_id, exc)
            return []
        messages = unwrap_messages(raw, self._file_log, "get_recent(tool)")
        candidates = collect_candidates(messages, limit)
        self._file_log.info(
            "工具召回：stream=%s 取到 %d 条消息 → 候选用户 %s",
            stream_id, len(messages), [f"{nick}({uid})" for uid, nick in candidates],
        )
        return candidates

    # -- 命令 ---------------------------------------------------------------

    @Command("ccr_debug_test", description="调试：在下一次 planner/reply 强制注入自己的私聊回忆（仅操作员）",
             pattern=r"(?<!\S)/?ccr\s+debug\s+test\s*$", permission="operator")
    async def cmd_ccr_debug_test(self, **kwargs: Any) -> tuple[bool, str, bool]:
        stream_id = str(kwargs.get("stream_id") or "")
        user_id = str(kwargs.get("user_id") or "").strip()
        if not stream_id or not user_id:
            return False, "无法识别当前聊流或命令发起者", True

        # 私聊流里 stream_id 对应的是私聊会话，注入逻辑同样适用（调试用）
        nickname = ""
        stream = await self.get_stream(stream_id)
        if stream:
            nickname = str(stream.get("user_nickname") or "").strip()

        self._set_force_inject(stream_id, user_id)

        row = self.store.get(user_id)
        if row and str(row.get("summary") or "").strip():
            text = (
                f"[ccr debug] 已标记：当前聊流的下一次 planner 与 reply 请求将强制注入"
                f"你（{nickname or user_id}）的私聊回忆。有效期 {int(_FORCE_INJECT_TTL // 60)} 分钟。"
            )
        else:
            # 暂无总结：立即定向扫描一次，标记保留至注入成功
            self._file_log.info("强制注入：用户 %s 暂无总结，触发定向扫描 stream=%s", user_id, stream_id)
            self._schedule_scan(stream_id, focus_user_ids=[user_id], reason="强制注入前定向扫描")
            text = (
                f"[ccr debug] 你（{nickname or user_id}）当前没有可用的私聊总结，"
                f"已触发一次定向扫描；总结完成后，当前聊流的下一次 planner 与 reply 请求将强制注入。"
            )
        await self.ctx.send.text(text, stream_id)
        return True, text, True

    @Command("ccr_clean", description="清空本插件的私聊回忆总结库（仅操作员）",
             pattern=r"(?<!\S)/?ccr\s+clean\s*$", permission="operator")
    async def cmd_ccr_clean(self, **kwargs: Any) -> tuple[bool, str, bool]:
        stream_id = str(kwargs.get("stream_id") or "")
        deleted = self.store.clear()
        # 顺带清掉所有强制注入与粘性召回标记，避免指向已被清空的数据
        self._force_inject.clear()
        self._tool_recall_sticky.clear()
        self._file_log.info("命令 /ccr clean 执行：删除 %d 条总结（发起者=%s）",
                            deleted, kwargs.get("user_id"))
        text = f"[ccr] 私聊回忆库已清空（删除 {deleted} 条总结）。"
        await self.ctx.send.text(text, stream_id)
        return True, text, True

    # -- Hook：消息推进 -----------------------------------------------------

    @HookHandler(_RECEIVE_HOOK, name="ccr_push_counter", mode=HookMode.OBSERVE, order=HookOrder.NORMAL)
    async def on_message_after_process(self, **kwargs: Any) -> None:
        cfg = self.config
        if not (cfg.plugin.enabled and cfg.scan.trigger_mode == "push"):
            return
        message = kwargs.get("message")
        if not isinstance(message, Mapping):
            return
        fields = extract_message_fields(message)
        # 只统计群聊里的真实用户消息（不含 bot 自身、通知、命令、注入合成记录）
        if not fields.get("group_id") or fields.get("is_self") or fields.get("is_synthetic"):
            return
        if fields.get("is_command"):
            return
        if not self._group_allowed(fields["group_id"]):
            return
        session_id = fields.get("session_id") or ""
        if not session_id:
            return
        count = self._push_counters.get(session_id, 0) + 1
        threshold = int(cfg.scan.scan_count)
        if count < threshold:
            self._push_counters[session_id] = count
            self._file_log.debug("消息推进：群 %s 计数 %d/%d", fields["group_id"], count, threshold)
            return
        self._push_counters[session_id] = 0
        self._file_log.info(
            "消息推进：群 %s 已推进 %d 条非 bot 消息（session=%s）→ 触发扫描", fields["group_id"], threshold, session_id,
        )
        self._schedule_scan(session_id, reason="消息推进")


def create_plugin() -> CrossChatRecallPlugin:
    """Runner 加载入口。"""
    return CrossChatRecallPlugin()
