from pynput.keyboard import Key, Controller
keyboard = Controller()
keyboard.press(Key.ctrl)
keyboard.press(Key.alt)
keyboard.press('t')
keyboard.release(Key.ctrl)
keyboard.release(Key.alt)
keyboard.release('t')
