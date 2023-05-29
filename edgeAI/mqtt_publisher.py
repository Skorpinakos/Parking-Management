# python3.6

import random

from paho.mqtt import client as mqtt_client





def connect_mqtt():

    broker = 'iot.patras5g.eu'
    port = 1872
    client_id = f'python-mqtt-{random.randint(0, 100)}'

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client,topic,msg):

    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")





