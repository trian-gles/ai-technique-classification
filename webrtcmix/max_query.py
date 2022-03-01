from pythonosc import udp_client

class Client:
    def __init__(self, ip: str):
        self._client = udp_client.SimpleUDPClient(ip, 5432)

    def send_arps(self, f1: float, f2: float, f3: float):
        self._client.send_message("/arps", [f1, f2, f3])


if __name__ == "__main__":
    c = Client("127.0.0.1")
    c.send_arps(440, 880, 1220)