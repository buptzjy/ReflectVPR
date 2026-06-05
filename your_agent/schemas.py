# schemas.py
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum
from PIL import Image

# =====================================================================
# 1. 路由选择的“路由标识”枚举
# =====================================================================
class RouteType(str, Enum):
    """
    大模型对图片任务分类的真实路由标识。
    """
    GLOBAL = "global"  # 仅天气/全局变化 -> IC-Light
    LOCAL = "local"    # 仅遮挡/局部变化 -> LightX2V
    DUAL = "dual"      # 天气+遮挡 -> LightX2V
    SKIP = "skip"


# =====================================================================
# 2. 路由模块（router.py）接收并解析的契约结构
# =====================================================================
class RouteDecision(BaseModel):
    """
    LLM 对输入图片进行多维度评估后，返回的结构化路由决策。
    字段完全对齐真实大模型返回的 JSON 格式，支持自动验证与转换。
    """
    file_name: str = Field(description="输入的图片文件名")
    city: str = Field(description="图片对应的城市")
    route: RouteType = Field(description="路由策略类型: global | local | dual | skip")
    
    # 多维度打分（满分各 10 分）
    weather_score: int = Field(ge=0, le=10, description="天气匹配得分（0~10）")
    occlusion_score: int = Field(ge=0, le=10, description="遮挡程度/复杂网络得分（0~10）")
    total_score: int = Field(ge=0, le=20, description="多维度评估总分（0~20）")
    
    # 模型选择（根据业务要求，后两者统一由 LightX2V 处理）
    candidate_model: str = Field(description="大模型推荐的候选模型")
    selected_model: Literal["IC-Light", "LightX2V-Local", "LightX2V-Dual", "None", "PENDING"] = Field(description="最终选定的图像生成引擎")
    
    # 决策原因
    reason: str = Field(description="LLM 做出该路由选择的详细文本依据")


# =====================================================================
# 3. 生成模块（generators/）的输入与输出契约
# =====================================================================
class GeneratorInput(BaseModel):
    """
    分发给底层具体生成器（IC-Light 或 LightX2V）的标准入参包装盒
    """
    image_path: str = Field(description="待处理的原图本地路径或服务器存储路径")
    prompt: str = Field(description="当前轮次指导光影生成的提示词（可能已被优化）")


class GeneratorOutput(BaseModel):
    """
    生成器引擎（不管走哪条路径）执行完毕后，必须返回的标准产物格式
    """
    file_path: str = Field(description="生成出来的结果图片或视频的最终保存路径")
    generator: Literal["IC-Light", "LightX2V"] = Field(description="标记是由哪个引擎实际生成的")
    metadata: dict = Field(default_factory=dict, description="扩展元数据，如 seed, cost_time 等参数")


# =====================================================================
# 4. 评估模块（evaluator.py）的打分反思契约
# =====================================================================
class EvalScore(BaseModel):
    """
    当前 Agent 使用的 Dual-Trait 评估结果。
    """
    s_geo: float = Field(ge=0, le=1, description="几何一致性得分 [0, 1]")
    s_div: float = Field(ge=0, le=1, description="多样性得分 [0, 1]")
    geo_ok: bool = Field(description="几何一致性是否通过")
    div_ok: bool = Field(description="多样性是否通过")
    passed: bool = Field(description="终止判据：双指标是否都通过")
    skipped: bool = Field(default=False, description="pass 路由是否跳过评估")
    feedback: dict = Field(default_factory=dict, description="传给 PromptRefiner 的反馈信息")


@dataclass
class AgentResult:
    final_image: Image.Image
    route: str
    rounds_used: int
    final_prompt: str
    final_score_geo: float
    final_score_div: float
    passed: bool


# =====================================================================
# 5. Agent 全局状态黑匣子（整个状态机的核心存储）
# =====================================================================
class AgentState(BaseModel):
    """
    主循环状态机。它像记忆体一样，贯穿并记录当前任务从头到尾的所有细节。
    """
    iteration: int = 0         # 当前正在进行第几轮循环（0 代表第一轮）
    max_iterations: int = 5    # 最大容忍的循环次数（防止进入无限死循环）
    
    image_path: str            # 初始输入的任务图片路径
    original_prompt: str       # 初始用户输入的、最原始的提示词（保持不变以备参考）
    current_prompt: str        # 当前轮次正在使用的提示词（每一轮被 Refiner 优化后，这个值会被覆写）
    
    route: Optional[str] = None  # global/local/dual，第一步确定后在后续整个生命周期中共享
    
    outputs: list[GeneratorOutput] = Field(default_factory=list)    # 历史生成产物列表
    scores: list[EvalScore] = Field(default_factory=list)           # 历史评估打分列表
    
    best_output: Optional[GeneratorOutput] = None  # 整个生命周期中，得分最高的那一次生成产物
    done: bool = False         # 整个 Agent 任务是否宣告结束（满足 passed=True 或达到最大迭代次数）
