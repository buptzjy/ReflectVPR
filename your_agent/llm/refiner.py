import os


class PromptRefiner:
    def __init__(self, llm_client):
        self.client = llm_client
        self.model = os.getenv(
            "REFLECTVPR_PLANNER_MODEL",
            "qwen3-vl-4b-instruct-remote",
        )

    def refine(self, old_prompt: str, feedback: dict, round_num: int) -> str:
        geo_issue = feedback.get("geo_issue", "")
        div_issue = feedback.get("div_issue", "")
        if isinstance(geo_issue, dict):
            geo_issue = (
                f"{geo_issue.get('prompt_instruction', '')} "
                f"参数建议: {geo_issue.get('param_adjustment', {})}"
            ).strip()
        if isinstance(div_issue, dict):
            div_issue = (
                f"{div_issue.get('prompt_instruction', '')} "
                f"参数建议: {div_issue.get('param_adjustment', {})}"
            ).strip()

        intensity_hint = {
            1: "做轻微调整",
            2: "做中等程度调整",
            3: "做较大幅度调整，确保满足要求",
        }.get(round_num, "做必要调整")

        issues = []
        if geo_issue:
            issues.append(f"【几何一致性问题】{geo_issue}")
        if div_issue:
            issues.append(f"【多样性问题】{div_issue}")
        issues_text = "\n".join(issues) if issues else "整体质量需要提升"

        refine_request = f"""你是一个图像生成prompt优化专家。

当前prompt（第{round_num}轮优化）：
{old_prompt}

评估发现的问题：
{issues_text}

请{intensity_hint}，重写一个改进版的英文prompt。

优化规则：
- 如果几何一致性不足：增加 "preserve exact road layout", "maintain original perspective", "keep lane markings intact" 等约束词
- 如果多样性不足：增强天气强度描述（如 heavy → torrential）或遮挡显著性（如 partial → dense occlusion）
- 保持prompt长度在50-80词
- 只返回新的prompt文字，不要解释。"""

        new_prompt = self.client.chat(
            system="你是一个图像生成prompt优化专家。",
            user=refine_request,
            temperature=0.6 + round_num * 0.1,
            model=self.model,
        )

        print(f"[Refiner] Round {round_num} 新Prompt: {new_prompt}")
        return new_prompt


if __name__ == "__main__":
    print("[Refiner] 模块加载成功，需传入llm_client才能运行。")
