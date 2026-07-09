"""Small reusable prompts used by local demonstrations and tests."""

from typing import Literal

from pydantic import BaseModel, Field

from enterprise_ai_assistant.prompt.template import FewShotExample, PromptTemplate


class SupportTicketClassification(BaseModel):
    """Structured classification result for one support question."""

    category: Literal["售前", "售后", "其他"] = Field(description="工单所属业务阶段")
    urgency: Literal["低", "中", "高"] = Field(description="建议处理优先级")
    reason: str = Field(min_length=1, description="分类理由")


def build_support_classification_prompt() -> PromptTemplate:
    """Build the first version of the support classification prompt."""

    return PromptTemplate(
        name="support-ticket-classification",
        version=1,
        system_template=(
            "你是企业客服工单分类助手。请依据示例判断业务阶段和处理优先级, "
            "不要补充用户没有提供的事实。"
        ),
        few_shot_examples=(
            FewShotExample(
                user="你们的企业版怎么收费?",
                assistant=(
                    '{"category":"售前","urgency":"低",'
                    '"reason":"用户正在咨询产品价格"}'
                ),
            ),
            FewShotExample(
                user="已经付款但服务无法使用, 影响线上业务。",
                assistant=(
                    '{"category":"售后","urgency":"高",'
                    '"reason":"已购服务故障并影响线上业务"}'
                ),
            ),
        ),
        user_template="请分类以下工单: {question}",
    )
