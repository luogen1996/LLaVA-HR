import requests

def get_img_question(server_ip,server_port,index):
    #http://{server_ip}:{server_port}/get_data?index={index}
    url = f"http://{server_ip}:{server_port}/get_data?index={index}"
    data = requests.get(url).json()
    img_path = data['img_path']
    question = data['question']
    question_id = data['question_id']
    return img_path,question,question_id