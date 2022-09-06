from re import L
import win32api
import win32con
import time
import keyboard

class AgentController:

    def __init__(self):
        self.x = 2560
        self.y = 1440
        self.key_map = {
            0: 0x57,
            1: 0x41,
            2: 0x53,
            3: 0x44
        }

    def action(self, input):
        print(input)
        button_press = [input[0], input[1], input[2], input[3]]
        self.move_mouse(input[4], input[5], input[6])
        self.keyboard(button_press)

    def move_mouse(self, mouse_click, mouse_x, mouse_y):
        #print(mouse_x, mouse_y)
        #win32api.SetCursorPos((int(self.x * mouse_x), int(self.y * mouse_y)))
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(0.7 * self.x * mouse_x), int(0.3 * self.y * mouse_y))
        if mouse_click == 1:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
            #time.sleep(.05)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)

    def keyboard(self, keys):
        for i, key in enumerate(keys):
            if key == 1:
                if i == 0:
                    keyboard.press('w')
                    #time.sleep(0.05)
                    #keyboard.release('w')
                elif i == 1:
                    keyboard.press('a')
                    #time.sleep(0.05)
                    #keyboard.release('a')
                elif i == 2:
                    keyboard.press('s')
                    #time.sleep(0.05)
                    #keyboard.release('s')
                elif i == 3:
                    keyboard.press('d')
                    #time.sleep(0.05)
                    #keyboard.release('d')
            else:
                if i == 0:
                    keyboard.release('w')
                elif i == 1:
                    keyboard.release('a')
                elif i == 2:
                    keyboard.release('s')
                elif i == 3:
                    keyboard.release('d')