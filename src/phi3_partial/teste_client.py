from phi3_partial.raw_server.partial_forward_client import PartialForwardClient

initial_input = {'dados': ''}


if __name__ == '__main__':

    client = PartialForwardClient([
        ('127.0.0.1', 9000)
    ])

    final_output = client.send_partial_forward(initial_input)

    print(final_output)

