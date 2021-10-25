import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Union
from webrtcmix.web_request import webrtc_request, play_np
from audiolazy import lazy_midi
from random import randrange, getrandbits, uniform, choice, seed

with open("blank_all_features.sco") as f:
    empty_score = f.read()




def insert_param(score: str, index_string: str, insert_string: str):
    index_string += " ="
    i = score.index(index_string) + len(index_string)
    return score[:i] + insert_string + score[i:]

def insert_under_comment(score: str, comment_string: str, insert_string: str):
    i = score.index(comment_string) + len(comment_string) + 2
    return score[:i] + insert_string + score[i:]

class Node:
    container = None
    def __init__(self, pitch: Union[str, float], wait: float = 1, dur: float = 1, left_max: int = 20,
                 left_edge = None, right_min: int = 0, right_edge = None):
        self.pitch = pitch
        self.wait = wait
        self.dur = dur
        self.left_max = left_max
        self.left_edge = left_edge
        self.right_min = right_min
        self.right_edge = right_edge
        #self.container.add_node(self)


class TreeContainer:
    """TODO: make nodes with same names doable.  Also make cursors copy at certain depths"""
    def __init__(self, nodes: List[Node] = None):
        self.G = nx.DiGraph()
        self.nodes = nodes if nodes else []

        self.params = {
        "max_dur": 30,
        "tempo": 100,
        "max_depth": 19,
        "max_curs": 1000,
        "init_curs": 1,
        "limit_depths": {},
        "limit_max_cursor_nums": {},
        "hardness": 8,
        "upper_decay": 1,
        "gliss_mode": "'uncoor'",
        "coor_stop_time": 20
        }

    def merge_with_other(self, other_cont, right_gate_delay: int = None, left_gate_max: int = None) -> bool:
        unlinked_node = self.find_unlinked_right_node()

        if left_gate_max:
            for n in self.nodes:
                n.left_max = left_gate_max

        if unlinked_node:
            unlinked_node.right_edge = other_cont.nodes[0].pitch
            self.nodes += other_cont.nodes.copy()
            if right_gate_delay:
                unlinked_node.right_min = right_gate_delay

        else:
            if right_gate_delay:
                return False
            unlinked_node = self.find_unlinked_left_node()
            if unlinked_node:
                unlinked_node.left_edge = other_cont.nodes[0].pitch
                self.nodes += other_cont.nodes

            else:
                return False
        return True



    def find_unlinked_right_node(self):
        for n in self.nodes:
            if not n.right_edge:
                return n

    def find_unlinked_left_node(self):
        for n in self.nodes:
            if not n.left_edge:
                return n


    def rand_graph_intervals(self, intervals: List[int], num_nodes: int = 7, pitches=("A4",)):
        container = TreeContainer()
        Node.index = 0
        Node.container = container
        more_pitches = [lazy_midi.str2midi(pitches[0])]
        if len(intervals) < 2:
            print("Need more intervals")

        while len(more_pitches) < num_nodes:  # fix this so intervals are built on old pitches
            i = 0
            print("looping")
            more_pitches_count = len(more_pitches)
            while i <= more_pitches_count:
                old_pitch = more_pitches[i]

                add = choice(intervals)
                if getrandbits(1):
                    add *= -1
                new_pitch = old_pitch + add
                if not new_pitch in more_pitches:
                    more_pitches.append(new_pitch)

                i += 1
        print("pitches finished")

        if len(more_pitches) > num_nodes:
            more_pitches = more_pitches[:num_nodes]
        for pitch in more_pitches:
            self.nodes.append(Node(lazy_midi.midi2str(pitch), wait=uniform(0.5, 5), dur=uniform(1, 4)))
        self.make_random_connections()
        self.make_all_connected()


    def make_random_connections(self):
        for n in self.nodes:
            possible_edges = [node.pitch for node in self.nodes if node.pitch != n.pitch]
            if getrandbits(1):
                n.right_edge = choice(possible_edges)
                possible_edges.remove(n.right_edge)
            n.left_edge = choice(possible_edges) if getrandbits(1) else None

    def add_node(self, node: Node):
        self.nodes.append(node)

    def visualize(self):
        edges = []
        for n in self.nodes:
            if n.left_edge:
                edges.append((n.pitch, n.left_edge))
            if n.right_edge:
                edges.append((n.pitch, n.right_edge))

        wait_times = [n.wait for n in self.nodes]
        max_wait = max(wait_times)
        colors = [t / max_wait for t in wait_times]
        colors = [0.5 for _ in wait_times]


        self.G.add_edges_from(edges)
        colors = ['yellow'] + ['pink' for _ in range(len(self.G.nodes) - 1)]

        pos = nx.circular_layout(self.G)
        nx.draw_networkx(self.G, pos, arrows=True, cmap = plt.get_cmap('jet'), node_size=500, node_color = colors)
        plt.ion()
        plt.draw()
        plt.pause(0.001)

    def hide(self):
        plt.close()

    def check_all_connected(self) -> List[Node]:
        connected_nodes = []

        def traverse_tree(n: Node):
            if n in connected_nodes:
                return
            connected_nodes.append(n)
            if n.left_edge:
                traverse_tree(self.get_node_by_pitch(n.left_edge))
            if n.right_edge:
                traverse_tree(self.get_node_by_pitch(n.right_edge))

        traverse_tree(self.nodes[0])

        connected_pitches = [n.pitch for n in connected_nodes]
        print(connected_pitches)
        return connected_nodes

    def make_all_connected(self):
        """TODO: fix this to guarantee connections.  its not hard"""
        already_connected = self.check_all_connected()
        all_nodes = set(self.nodes)

        unconnected = all_nodes.difference(set(already_connected))

        delete_nodes = []

        for unc_node in unconnected: # you can fix this pretty easily with i += 1
            connected = False
            for con_node in already_connected:
                if not con_node.left_edge and con_node.right_edge != unc_node.pitch:
                    con_node.left_edge = unc_node.pitch
                    connected = True
                    break
                if not con_node.right_edge and con_node.left_edge != unc_node.pitch:
                    con_node.right_edge = unc_node.pitch
                    connected = True
                    break
            if not connected:
                print(f"can't connect {unc_node.pitch}")
                delete_nodes.append(unc_node)



        loop = self.check_loop()
        if not loop:
            try:
                self.make_all_connected() # repeat it till theres a loop!!
            except RecursionError:
                choice(self.nodes[1:]).right_edge = self.nodes[0].pitch
                for n in delete_nodes:
                    self.remove_node(n)

    def remove_node(self, rm_node: Node):
        """TODO needs an additional setting for repairing a broken graph"""
        self.nodes.remove(rm_node)
        for node in self.nodes:
            if node.right_edge == rm_node.pitch:
                node.right_edge = None
            if node.left_edge == rm_node.pitch:
                node.left_edge = None

    def check_loop(self):
        """Simply checks if there are more steps than nodes"""
        self.loop = False

        def check_loop(n: Node, depth: int):
            depth += 1
            if depth > len(self.nodes):
                self.loop = True
                return
            if n.right_edge:
                rn = self.get_node_by_pitch(n.right_edge)
                check_loop(rn, depth)
            if n.left_edge:
                ln = self.get_node_by_pitch(n.left_edge)
                check_loop(ln, depth)

        check_loop(self.nodes[0], 0)

        return self.loop

    def get_node_by_pitch(self, pitch: Union[str, float]):
        for n in self.nodes:
            if n.pitch == pitch:
                return n

    def index_of_node_by_pitch(self, pitch: Union[str, float]):
        for n in self.nodes:
            if pitch == n.pitch:
                return self.nodes.index(n)
        return -1

    def get_rtc_score(self):
        node_make_str = ""
        for n in self.nodes:
            midi_pitch = lazy_midi.str2midi(n.pitch)
            left_index = self.index_of_node_by_pitch(n.left_edge)
            right_index = self.index_of_node_by_pitch(n.right_edge)

            node_make_str += f"make_note_node({n.wait}, {n.dur}, {midi_pitch}, {n.left_max}, {left_index}, {n.right_min}, {right_index})\n"

        rtc_score = insert_under_comment(empty_score, "BUILD HERE", node_make_str)
        for param, val in self.params.items():
            rtc_score = insert_param(rtc_score, param, str(val))

        #print(rtc_score)

        return rtc_score

    def listen(self):
        rtc_score = self.get_rtc_score()
        play_np(webrtc_request(rtc_score))


saved_nodes = {}
saved_containers = {}

def save_nodes(cont: TreeContainer, name: str):
    saved_nodes[name] = cont.nodes.copy()

def save_container(cont: TreeContainer, name: str):
    saved_containers[name] = cont




def test_rand():
    print("Testing random graph")
    container = TreeContainer()
    Node.container = container
    container.rand_graph_intervals(intervals=[3, 7], num_nodes=6)
    print(container.check_loop())
    container.visualize()
    container.listen()
    with open("last_score.sco", "w") as f:
        f.write(container.get_rtc_score())

def test_merge():
    seed(3) # 3 works, also 9
    print("Testing random graph")
    container = TreeContainer()
    container.rand_graph_intervals(intervals=[3, 7], num_nodes=4)
    # container.visualize()
    # container.listen()
    # container.hide()

    other_container = TreeContainer()
    other_container.rand_graph_intervals(intervals=[2, 5], num_nodes=4, pitches=("F6",))
    # other_container.visualize()
    # other_container.listen()
    # other_container.hide()
    if container.merge_with_other(other_container, 6, 8):
        container.visualize()
        container.listen()
    else:
        print("Unlinkable")



def test_preset():
    container = TreeContainer()
    Node.container = container
    e3 = Node("D3", wait=0.7, dur=1, left_edge="B2", left_max=23)  # 0
    b2 = Node("B2", wait=1.11, left_edge="F#5", left_max=15, right_edge="G5", right_min=5)  # 1
    fs5 = Node("F#5", wait=0.51, dur=0.5, left_max=17, left_edge="D3", right_min=1, right_edge="G5")  # 2
    g5 = Node("G5", wait=2.82, dur=2.2, left_max=16, left_edge="F#5", right_edge="G4", right_min=6)  # 3
    a5 = Node("A5", wait=7, dur=2.2, left_max=12, right_min=13, left_edge="A3", right_edge="D5")  # 4
    d5 = Node("D5", wait=1, dur=4, left_max=23, right_min=18, left_edge="G4", right_edge="G5")  # 5
    g4 = Node("G4", wait=1, dur=4, left_max=23, right_min=16, left_edge="G3", right_edge="F#5")  # 6
    g3 = Node("G3", wait=5, dur=4, left_max=23, right_min=15, left_edge="G6", right_edge="G#6")  # 7
    g6 = Node("G6", wait=5, dur=4, left_max=20, right_min=8, left_edge="G4", right_edge="E2")  # 8
    e2 = Node("E2", wait=6, dur=4, right_min=8, right_edge="A3")  # 9
    a3 = Node("A3", wait=5, dur=4, left_max=23, left_edge="G#6")  # 10
    gs6 = Node("G#6", wait=5, dur=4)  # 11

    container.visualize()
    container.listen()


if __name__ == "__main__":
    test_merge()


