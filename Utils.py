import json
import os


def convert_file_to_json(file_path):
    # 打开文件
    with open(file_path, 'r') as file:
        # 读取所有行
        lines = file.readlines()

    # 创建一个空列表来存储json对象
    json_list = []

    # 遍历每一行
    for line in lines:
        # 去掉行尾的换行符
        line = line.strip()
        # 将行转换成json对象
        json_obj = json.loads(line)
        # 将json对象添加到列表中
        json_list.append(json_obj)

    return json_list


def token_location_redirection(token_list, trigger, tokenizer):
    trigger_token = tokenizer.tokenize(trigger)
    for i in range(len(token_list) - len(trigger_token) + 1):
        if token_list[i:i + len(trigger_token)] == trigger_token:
            return i, i + len(trigger_token)
    return 0, 0


def get_event_schema(source_data_path, encode_data_path):
    if os.path.isfile(encode_data_path + "event_schema.json"):
        with open(encode_data_path + "event_schema.json", "r") as infile:
            event_schema = json.load(infile)
    else:
        event_schema = {}
        if not os.path.isfile(source_data_path + f"duee_schema/duee_event_schema.json"):
            raise FileNotFoundError("The target data cannot be generated because the source data does not exist")
        else:
            json_file = convert_file_to_json(source_data_path + f"duee_schema/duee_event_schema.json")
            for jsonl in json_file:
                event_schema[jsonl['event_type']] = [role['role'] for role in jsonl['role_list']]
            if not os.path.exists(encode_data_path):
                os.makedirs(encode_data_path)
            with open(encode_data_path + "event_schema.json", "w") as outfile:
                json.dump(event_schema, outfile)
    return event_schema