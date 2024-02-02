import requests


def get_model_and_data(task_name,server_ip,server_port):
    model_path = f"http://{server_ip}:{server_port}/io_info"
    data_path = f"http://{server_ip}:{server_port}/meta_info"
    model_info = requests.get(model_path).json()
    data_info = requests.get(data_path).json()
    checkpoint_path = model_info['checkpoint_path']
    output_dir = model_info['output_dir']
    dataset_name = data_info['name']
    dataset_length = data_info['length']
    dataset_type = data_info['type']
    return checkpoint_path, output_dir, dataset_name, dataset_length, dataset_type