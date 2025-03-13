import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

# 引入Agent与AgentChain
from agent.questionnaire_agent import QuestionnaireAgent
from agent.grading_agent import GradingAgent
from agent.agent_chain import AgentChain

# 设置日志记录
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # 输出到控制台
                        logging.FileHandler("app.log", mode='w')  # 输出到文件
                    ])
logger = logging.getLogger()

load_dotenv()  # 加载环境变量

app = Flask(__name__, static_folder='frontend')

# 读取 KEY 和 BASE_URL
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")  # 不设置则默认为官方地址

# 初始化Agent
q_agent = QuestionnaireAgent(api_key=API_KEY, base_url=BASE_URL, model="qwen-plus")
g_agent = GradingAgent(api_key=API_KEY, base_url=BASE_URL, model="qwen-plus")

# 组合到 AgentChain
chain = AgentChain(q_agent, g_agent)

# 用于存储question_type的全局字典
question_types_storage = {}

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/generate_questionnaire', methods=['POST'])
def api_generate_questionnaire():
    """
    生成完整的问卷。
    """
    try:
        data = request.get_json()
        long_text = data.get('long_text', '')
        num_questions_per_chunk = data.get('num_questions_per_chunk', 5)
        question_type = data.get('question_type')  # 从前端接收题型

        if not long_text:
            logger.error('缺少 long_text 参数')
            return jsonify({'error': '缺少 long_text 参数'}), 400

        # 存储question_type，以便后续使用
        question_types_storage['question_type'] = question_type

        # 调用 AgentChain 来生成问卷
        questions = chain.process_long_text(long_text, num_questions_per_chunk, question_type)
        logger.info(f"成功生成问卷：{len(questions)}个问题")
        return jsonify({'questions': questions})
    except Exception as e:
        logger.exception("生成问卷时发生错误")
        return jsonify({'error': '生成问卷时发生错误'}), 500

@app.route('/api/submit_answers', methods=['POST'])
def submit_answers():
    """
    处理用户提交的问卷答案。
    """
    try:
        data = request.get_json()
        answers = data.get('answers', [])
        questions = data.get('questions', [])

        if not answers or not questions:
            logger.error('缺少答案数据或题目数据')
            return jsonify({'error': '缺少答案数据或题目数据'}), 400

        # 获取之前存储的 question_type
        question_type = question_types_storage.get('question_type', '普通问答题')

        # 调用 AgentChain 来批改答案
        feedback, total_score = chain.grade_answers(questions, answers, question_type)

        logger.info(f"成功批改问卷，得分：{total_score}")
        return jsonify({
            'feedback': feedback,
            'total_score': total_score
        })
    except Exception as e:
        logger.exception("批改答案时发生错误")
        return jsonify({'error': '批改答案时发生错误'}), 500

if __name__ == "__main__":
    logger.info("启动 Flask 应用")
    app.run(debug=True, port=5000)
