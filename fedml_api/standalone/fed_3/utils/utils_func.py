def get_client_list(client_index, client_list):
    ret_client_list = []
    for index in client_index:
        ret_client_list.append(client_list[index])
    return ret_client_list


def get_total_training_intensity(clients):
    tot_training_intensity = 0
    for client in clients:
        tot_training_intensity += client.get_training_intensity()
    return tot_training_intensity

