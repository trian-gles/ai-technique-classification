class StackDict:
    def __init__(self, depth: int):
        self.list = []
        self.depth = depth

    def add(self, val: dict):
        self.list.insert(0, val)
        if len(self.list) > self.depth:
            self.list = self.list[:-2]

    def peek(self) -> dict:
        """Look at the top value"""
        return self.list[0]

    def filter_to_list(self, entry: str):
        return [note[entry] for note in self.list]


class Brain:
    def __init__(self):
        self.heat = 0
        self.prior_notes = StackDict(7)

    def new_note(self, note_dict: dict): # maybe only certain information should be stored about each note?
        if note_dict["prediction"][0] == "SILENCE":
            if self.prior_notes.peek()["prediction"][0] == "pont":
                print("Pont into Silence")

        self.prior_notes.add(note_dict)

