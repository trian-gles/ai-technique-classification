from pythonosc import udp_client

class Client:
    def __init__(self, ip: str):
        self._client = udp_client.SimpleUDPClient(ip, 5432)

    def send_arps(self, f1: float, f2: float, f3: float):
        self._client.send_message("/arps", [f1, f2, f3])

    def send_clar(self):
        self._client.send_message("/clar", 0.0)


if __name__ == "__main__":
    c = Client("192.168.1.3")
    c.send_arps(440, 880, 1220)
    c.send_clar()