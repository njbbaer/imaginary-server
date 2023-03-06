from flask import Flask, Response, request
from langchain.llms import OpenAIChat
from langchain import PromptTemplate, LLMChain
import yaml
import re

app = Flask(__name__)


def read_prompts_file():
    with open('prompts.yml', 'r') as file:
        return yaml.safe_load(file)


def create_llm_chain():
    prompts_file = read_prompts_file()
    prefix_messages = [{'role': 'system', 'content': prompts_file['system']}]
    llm = OpenAIChat(temperature=0, stop='<|end|>', prefix_messages=prefix_messages)
    prompt = PromptTemplate(template=prompts_file['user'], input_variables=["raw_request"])
    return LLMChain(prompt=prompt, llm=llm)


def run_llm_chain(raw_request):
    llm_chain = create_llm_chain()
    return llm_chain.run(raw_request)


def extract_response_code(raw_response):
    match = re.search(r'\d{3}', raw_response)
    if match:
        return match.group(0)
    else:
        return 200


def extract_content_type(raw_response):
    match = re.search(r'Content-Type: (.+)', raw_response)
    if match:
        return match.group(0)
    else:
        return 'application/json'


def extract_raw_request(request):
    raw_request = request.method + ' ' + request.url + ' ' + request.environ['SERVER_PROTOCOL'] + '\n'
    for header in request.headers:
        raw_request += header[0] + ': ' + header[1] + '\n'
    raw_request += '\n' + request.get_data(as_text=True)
    return raw_request


def extract_raw_response(full_response):
    start_index = full_response.index('<|start|>') + 9
    if start_index == -1:
        raise Exception('Could not find start tag in LLM response.')
    return full_response[start_index:]


def extract_response_body(raw_response):
    start_index = raw_response.find('\n\n') + 2
    response_body = raw_response[start_index:]
    return response_body


@app.route('/<path:path>', methods=['GET', 'HEAD', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'])
def catch_all(path):
    raw_request = extract_raw_request(request)
    full_response = run_llm_chain(raw_request)
    print('#' * 80)
    print(full_response)
    raw_response = extract_raw_response(full_response)
    response_code = extract_response_code(raw_response)
    response_body = extract_response_body(raw_response)
    content_type = extract_content_type(raw_response)
    return Response(
        response_body,
        status=response_code,
        headers={'Content-Type': content_type}
    )
