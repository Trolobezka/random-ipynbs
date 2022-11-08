from typing import Dict, List, Tuple, Union


class Pero(object):
    @classmethod
    def rozmery_podle_prumeru(cls) -> Dict[Tuple[int, int], Dict[str, float]]:
        return {
            (6, 8): {"b": 2.0, "h": 2.0, "t": 1.1, "t1": 0.9},
            (8, 10): {"b": 3.0, "h": 3.0, "t": 1.7, "t1": 1.3},
            (10, 12): {"b": 4.0, "h": 4.0, "t": 2.4, "t1": 1.6},
            (12, 17): {"b": 5.0, "h": 5.0, "t": 2.9, "t1": 2.1},
            (17, 22): {"b": 5.0, "h": 6.0, "t": 3.5, "t1": 2.5},
            (22, 30): {"b": 8.0, "h": 7.0, "t": 4.1, "t1": 2.9},
            (30, 38): {"b": 10.0, "h": 8.0, "t": 4.7, "t1": 3.3},
            (38, 44): {"b": 12.0, "h": 8.0, "t": 4.9, "t1": 3.1},
            (44, 50): {"b": 14.0, "h": 9.0, "t": 5.5, "t1": 3.5},
            (50, 58): {"b": 16.0, "h": 10.0, "t": 6.2, "t1": 3.8},
            (58, 65): {"b": 18.0, "h": 11.0, "t": 6.8, "t1": 4.2},
            (65, 75): {"b": 20.0, "h": 12.0, "t": 7.4, "t1": 4.6},
            (75, 85): {"b": 22.0, "h": 14.0, "t": 8.5, "t1": 5.5},
            (85, 95): {"b": 25.0, "h": 14.0, "t": 8.7, "t1": 5.3},
            (95, 110): {"b": 28.0, "h": 16.0, "t": 9.9, "t1": 6.1},
            (110, 130): {"b": 28.0, "h": 18.0, "t": 11.1, "t1": 6.9},
            (130, 150): {"b": 36.0, "h": 20.0, "t": 12.3, "t1": 7.7},
            (150, 170): {"b": 40.0, "h": 22.0, "t": 13.5, "t1": 8.5},
            (170, 200): {"b": 40.0, "h": 25.0, "t": 15.3, "t1": 9.7},
            (200, 230): {"b": 45.0, "h": 28.0, "t": 17.0, "t1": 11.0},
            (230, 260): {"b": 50.0, "h": 32.0, "t": 19.3, "t1": 12.7},
            (260, 290): {"b": 63.0, "h": 32.0, "t": 19.6, "t1": 12.4},
            (290, 330): {"b": 70.0, "h": 36.0, "t": 22.0, "t1": 14.0},
            (330, 380): {"b": 80.0, "h": 40.0, "t": 24.6, "t1": 15.4},
        }

    @classmethod
    def normalizovane_delky(cls) -> List[int]:
        return [
            *range(2, 22, 2),
            22,
            25,
            *range(28, 40, 4),
            40,
            45,
            50,
            63,
            *range(70, 110, 10),
            110,
            125,
            *range(140, 220, 20),
            220,
            250,
            280,
            315,
            355,
            400,
        ]


class RovnobokeDrazkovani(object):
    @classmethod
    def rozmery(cls) -> List[Dict[str, Union[str, float]]]:
        return [
            {"rada": "lehka", "d": 23, "n": 6.0, "D": 26.0, "b": 6.0},
            {"rada": "lehka", "d": 26, "n": 6.0, "D": 30.0, "b": 6.0},
            {"rada": "lehka", "d": 28, "n": 6.0, "D": 32.0, "b": 7.0},
            {"rada": "lehka", "d": 32, "n": 8.0, "D": 36.0, "b": 6.0},
            {"rada": "lehka", "d": 36, "n": 8.0, "D": 40.0, "b": 7.0},
            {"rada": "lehka", "d": 42, "n": 8.0, "D": 46.0, "b": 8.0},
            {"rada": "lehka", "d": 46, "n": 8.0, "D": 50.0, "b": 9.0},
            {"rada": "lehka", "d": 52, "n": 8.0, "D": 58.0, "b": 10.0},
            {"rada": "lehka", "d": 56, "n": 8.0, "D": 62.0, "b": 10.0},
            {"rada": "lehka", "d": 62, "n": 8.0, "D": 68.0, "b": 12.0},
            {"rada": "lehka", "d": 72, "n": 10.0, "D": 78.0, "b": 12.0},
            {"rada": "lehka", "d": 82, "n": 10.0, "D": 88.0, "b": 12.0},
            {"rada": "lehka", "d": 92, "n": 10.0, "D": 98.0, "b": 14.0},
            {"rada": "lehka", "d": 102, "n": 10.0, "D": 108.0, "b": 16.0},
            {"rada": "lehka", "d": 112, "n": 10.0, "D": 120.0, "b": 18.0},
            {"rada": "stredni", "d": 11, "n": 6.0, "D": 14.0, "b": 3.0},
            {"rada": "stredni", "d": 13, "n": 6.0, "D": 16.0, "b": 3.5},
            {"rada": "stredni", "d": 16, "n": 6.0, "D": 20.0, "b": 4.0},
            {"rada": "stredni", "d": 18, "n": 6.0, "D": 22.0, "b": 5.0},
            {"rada": "stredni", "d": 21, "n": 6.0, "D": 25.0, "b": 5.0},
            {"rada": "stredni", "d": 23, "n": 6.0, "D": 28.0, "b": 6.0},
            {"rada": "stredni", "d": 26, "n": 6.0, "D": 32.0, "b": 6.0},
            {"rada": "stredni", "d": 28, "n": 6.0, "D": 34.0, "b": 7.0},
            {"rada": "stredni", "d": 32, "n": 8.0, "D": 38.0, "b": 6.0},
            {"rada": "stredni", "d": 36, "n": 8.0, "D": 42.0, "b": 7.0},
            {"rada": "stredni", "d": 42, "n": 8.0, "D": 48.0, "b": 8.0},
            {"rada": "stredni", "d": 46, "n": 8.0, "D": 54.0, "b": 9.0},
            {"rada": "stredni", "d": 52, "n": 8.0, "D": 60.0, "b": 10.0},
            {"rada": "stredni", "d": 56, "n": 8.0, "D": 65.0, "b": 10.0},
            {"rada": "stredni", "d": 62, "n": 8.0, "D": 72.0, "b": 12.0},
            {"rada": "stredni", "d": 72, "n": 10.0, "D": 82.0, "b": 12.0},
            {"rada": "stredni", "d": 82, "n": 10.0, "D": 92.0, "b": 12.0},
            {"rada": "stredni", "d": 92, "n": 10.0, "D": 102.0, "b": 14.0},
            {"rada": "stredni", "d": 102, "n": 10.0, "D": 112.0, "b": 16.0},
            {"rada": "stredni", "d": 112, "n": 10.0, "D": 112.0, "b": 16.0},
        ]
