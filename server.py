from flask import Flask, Response, request
from langchain.llms import OpenAIChat
from langchain import PromptTemplate, LLMChain
import yaml
import re

app = Flask(__name__)


def create_llm_chain():
    prompts_file = read_prompts_file()
    prefix_messages = [{'role': 'system', 'content': prompts_file['system']}]
    llm = OpenAIChat(temperature=0, stop='<|end|>', prefix_messages=prefix_messages)
    prompt = PromptTemplate(template=prompts_file['user'], input_variables=['raw_request'])
    return LLMChain(prompt=prompt, llm=llm)


def read_prompts_file():
    with open('prompts.yml', 'r') as file:
        return yaml.safe_load(file)


def run_llm_chain(raw_request):
    llm_chain = create_llm_chain()
    return llm_chain.run(raw_request)


def convert_request_into_text(request):
    raw_request = f'{request.method} {request.url} {request.environ["SERVER_PROTOCOL"]}\n'
    for header in request.headers:
        raw_request += f'{header[0]}: {header[1]}\n'
    raw_request += '\n' + request.get_data(as_text=True)
    return raw_request


def extract_response_code(raw_response):
    match = re.search(r'\d{3}', raw_response)
    return match.group(0) if match else '200'


def extract_response_body(raw_response):
    start_index = raw_response.find('\n\n') + 2
    response_body = raw_response[start_index:]
    return response_body


def extract_headers(raw_response):
    headers = re.findall(r'^([\w-]+):\s*(.+)$', raw_response, re.MULTILINE)
    return {name: value for name, value in headers}


def extract_raw_response(llm_response):
    start_index = llm_response.index('<|start|>') + 9
    if start_index == -1:
        raise Exception('Missing <|start|> tag in LLM response.')
    return llm_response[start_index:]


def create_http_response(llm_response):
    raw_response = extract_raw_response(llm_response)
    response_code = extract_response_code(raw_response)
    response_body = extract_response_body(raw_response)
    headers = extract_headers(raw_response)
    return Response(
        response_body,
        status=response_code,
        headers=headers
    )


@app.route('/<path:path>', methods=['GET', 'HEAD', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'])
def catch_all(path):
    request_text = convert_request_into_text(request)
    llm_response = run_llm_chain(request_text)
    print(llm_response)
    return create_http_response(llm_response)
