# backend/agent/agent_chain.py

from agent.questionnaire_agent import QuestionnaireAgent
from agent.grading_agent import GradingAgent

class AgentChain:
    def __init__(self, q_agent: QuestionnaireAgent, g_agent: GradingAgent):
        self.q_agent = q_agent
        self.g_agent = g_agent

    def process_long_text(self, long_text: str, num_questions_per_chunk: int, question_type: str = "普通问答题"):
        """
        用问卷Agent生成题目
        """
        return self.q_agent.generate_questionnaire(
            long_text,
            num_questions_per_chunk=num_questions_per_chunk,
            question_type=question_type
        )

    def grade_answers(self, questions: list, answers: list, question_type: str):
        """
        依次调用GradingAgent对每道题进行打分
        """
        feedback = []
        total_score = 0
        for question, answer in zip(questions, answers):
            result = self.g_agent.grade_question(question, answer, question_type)
            score = result['score']
            total_score += score
            feedback.append({
                'question': question,
                'answer': answer,
                'score': score,
                'feedback': result['feedback']
            })
        return feedback, total_score
