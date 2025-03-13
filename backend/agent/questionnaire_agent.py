# backend/agent/questionnaire_agent.py

import openai
import re
import json
import tiktoken

class QuestionnaireAgent:
    def __init__(self, api_key: str, base_url: str = None, model: str = "qwen-plus"):
        """
        使用官方 openai 库进行初始化。
        如果需要自定义 base_url(例如兼容其他服务端), 可以赋值给 openai.api_base。
        """
        openai.api_key = api_key
        if base_url:
            openai.api_base = base_url
        self.model = model

    def split_text_by_tokens(self, text: str, max_tokens=2000) -> list:
        """
        使用 tiktoken 对文本进行 token 化，根据 max_tokens 切分成多个 chunk。
        每个 chunk decode 回原文本，以保证能够投喂到模型里。
        """
        # 根据self.model获取对应的tokenizer, 若不支持qwen-plus，可换成"gpt-3.5-turbo"的encoding
        # 或者使用 tiktoken.get_encoding("cl100k_base")
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except:
            # 若tiktoken不支持该模型，可以fallback到一个通用编码
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end
        return chunks


    def generate_questions_from_chunk(self, chunk: str, num_questions=5, question_type="普通问答题") -> str:
        """
        给定文本 chunk，生成若干问题。
        question_type: 前端传入的"普通问答题"、"选择题"、"判断题"等
        """
        try:
            prompt = (
                f"根据以下内容生成{num_questions}个'{question_type}'问题：\n\n{chunk}\n\n"
                f"要求：\n"
                f"1. 不要输出多余的话，不要给出答案。\n"
                f"2. 如果是选择题，把选项和题目写在同一行内；如果是判断题，注意只生成判断内容。\n"
                f"3. 输出时每个问题独立占一行。\n"
                f"4. 只出与核心思想和关键事件相关的问题，不要出无关数据或生僻数字作为考题，要注重对价值观的考察，而非过于具体的时间和数字。\n"
            )

            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are an expert in the history of the Communist Party of China.'},
                    {'role': 'user', 'content': prompt}
                ],
            )
            # 官方库的结构
            content = completion.choices[0].message.content

            # 去除**符号
            content = re.sub(r'\*\*', '', content)

            return content.strip()
        except Exception as e:
            print(f"Error generating questions for chunk: {e}")
            return ""

    def generate_questionnaire(self, long_text: str, num_questions_per_chunk=5, question_type="普通问答题") -> list:
        """
        主函数：先用token计数切分文本，再对每个chunk调用 generate_questions_from_chunk。
        question_type 由前端传入，默认为"普通问答题"。
        """
        chunks = self.split_text_by_tokens(long_text, max_tokens=2000)
        all_questions = []

        for idx, chunk in enumerate(chunks):
            print(f"Processing chunk {idx + 1}/{len(chunks)}")
            questions = self.generate_questions_from_chunk(chunk,
                                                           num_questions=num_questions_per_chunk,
                                                           question_type=question_type)
            if questions:
                # 按行拆分
                question_lines = re.split(r'\n+', questions)
                for line in question_lines:
                    clean_line = re.sub(r'^\d+\.?\s*', '', line).strip()
                    if clean_line:
                        all_questions.append(clean_line)

        # 去重
        unique_questions = list(dict.fromkeys(all_questions))
        return unique_questions