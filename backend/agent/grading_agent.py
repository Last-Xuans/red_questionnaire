import openai
import json
import re
import logging

# 设置日志
logger = logging.getLogger()

class GradingAgent:
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-3.5-turbo"):
        """
        使用官方 openai 库进行初始化。
        """
        openai.api_key = api_key
        if base_url:
            openai.api_base = base_url
        self.model = model

    def grade_question(self, question: str, answer: str, question_type: str) -> dict:
        """
        输入题目和答案，根据题目类型返回 {'score': int, 'feedback': str}
        """
        try:
            # 对于普通问答题
            if question_type == "普通问答题":
                prompt = (
                    f"题目：{question}\n"
                    f"答案：{answer}\n\n"
                    "请根据以下评分标准为此题打分，满分10分：\n"
                    "1. 内容准确性（4 分）：\n"
                    "    - 4 分：答案内容完全准确，涵盖了所有关键点。\n"
                    "    - 3 分：答案内容大部分准确，但有少数遗漏。\n"
                    "    - 2 分：答案存在较大误差，缺少关键点。\n"
                    "    - 1 分：答案内容严重偏离正确答案。\n"
                    "2. 逻辑清晰度（3 分）：\n"
                    "    - 3 分：答案条理清晰，逻辑严谨，易于理解。\n"
                    "    - 2 分：答案结构较清晰，逻辑大致合适，但部分地方表达不清。\n"
                    "    - 1 分：答案缺乏清晰的结构，逻辑混乱。\n"
                    "3. 语言表达（2 分）：\n"
                    "    - 2 分：语言流畅，表达清晰，几乎没有语法错误。\n"
                    "    - 1 分：语言基本流畅，存在一些语法错误，但不影响理解。\n"
                    "    - 0 分：语言表达较差，语法错误多，影响理解。\n"
                    "4. 完整性（1 分）：\n"
                    "    - 1 分：答案涵盖了所有要求的要点，完整回答问题。\n"
                    "    - 0 分：答案不完整，缺少重要信息。\n\n"
                    "请根据以上标准对这个答案进行评分，并给出详细的反馈。务必按以下格式返回：\n"
                    "score: X分\n"
                    "feedback: <这里写出详细评价，基于上述评分标准给出反馈>\n"
                    "不要输出其他内容。\n"
                )
            else:
                # 对于选择题和判断题，使用简单评分和反馈
                prompt = (
                    f"题目：{question}\n"
                    f"答案：{answer}\n\n"
                    "请为此题进行评分（满分10分）并给出简短的反馈。\n"
                    "只能给出10分或0分两种分数\n"
                    "返回格式：\n"
                    "score: X分\n"
                    "feedback: <简短的反馈说明>\n"
                    "不要输出其他内容。\n"
                )

            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are an expert in grading essay-type questions.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            content = completion.choices[0].message.content.strip()

            score_match = re.search(r"score:\s*(\d+)\s*分", content)
            feedback_match = re.search(r"feedback:\s*(.*)", content, re.DOTALL)

            score = 0
            feedback = ""
            if score_match:
                score = int(score_match.group(1))

            if feedback_match:
                feedback = feedback_match.group(1).strip()

            logger.info(f"评分：{score} 分，反馈：{feedback}")
            return {'score': score, 'feedback': feedback}

        except Exception as e:
            logger.exception("评分时发生错误")
            return {'score': 0, 'feedback': '评分出错'}
