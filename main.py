import cv2
from Virtualkeyboard.HandDetectionModule import HandDetectionModule
import math
import time

capture = cv2.VideoCapture(0)
detect = HandDetectionModule()

pressed_keys = []
capture.set(3, 1280)
capture.set(4, 720)

keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
    ["Space", "Backspace", "Enter"]
]


last_pressed_time = {}
cooldown_duration = 0.5

def findHands(image, hands):
    for hand in hands:
        label = hand['label']
        landmarks = hand['landmarks']

        lmList = []
        h, w, _ = image.shape
        x_list, y_list = [], []

        for id, lm in enumerate(landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
            x_list.append(cx)
            y_list.append(cy)

        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)

        # Draw black rectangle around hand
        cv2.rectangle(image, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 0, 0), 2)

        # Draw label text on top of bounding box
        cv2.putText(image, f"{label}", (x_min - 20, y_min - 25),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)


def draw_keyboard(image):
    key_height = 60
    offset_x = 300
    offset_y = 0
    spacing = 10
    key_boxes = []

    for i, row in enumerate(keys):
        current_x = offset_x
        y = offset_y + i * (key_height + spacing)

        for key in row:
            width = 60

            if key == "Space":
                width = 60 * 4 + spacing * 3
            elif key == "Backspace":
                width = 60 * 3 + spacing
            elif key == "Enter":
                width = 60 * 2 + spacing

            # Draw key
            cv2.rectangle(image, (current_x, y), (current_x + width, y + key_height), (255, 255, 255), 2)
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = current_x + (width - text_size[0]) // 2
            text_y = y + (key_height + text_size[1]) // 2
            cv2.putText(image, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            key_boxes.append((key, current_x, y, width, key_height))

            current_x += width + spacing

    return key_boxes


def rightHand(multiHandsData, image, key_boxes, left_hand_position):
    global pressed_keys

    if left_hand_position is None:
        return

    current_time = time.time()

    for hand_data in multiHandsData:
        hand_label = hand_data[0][0].capitalize()

        if hand_label == "Right":
            x1, y1 = hand_data[4][2], hand_data[4][3]
            x2, y2 = hand_data[8][2], hand_data[8][3]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw thumb and index on the image
            cv2.circle(image, (x1, y1), 10, (0, 0, 0), cv2.FILLED)
            cv2.circle(image, (x2, y2), 10, (0, 0, 0), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 3)
            cv2.circle(image, (cx, cy), 10, (0, 0, 0), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            if length < 30:
                cv2.circle(image, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

                for key, key_x, key_y, key_w, key_h in key_boxes:
                    if isKeyPressed((left_hand_position[0], left_hand_position[1]), (key_x, key_y, key_w, key_h)):

                        if key not in last_pressed_time or (current_time - last_pressed_time[key]) > cooldown_duration:
                            handleKeyPress(key)
                            last_pressed_time[key] = current_time


                        cv2.rectangle(image, (key_x, key_y), (key_x + key_w, key_y + key_h), (0, 0, 255), cv2.FILLED)
                        cv2.putText(image, key, (key_x + 10, key_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


def leftHand(multiHandsData, image, key_boxes):
    left_hand_position = None

    for hand_data in multiHandsData:
        hand_label = hand_data[0][0].capitalize()

        if hand_label == "Left":
            x1, y1 = hand_data[8][2], hand_data[8][3]

            left_hand_position = (x1, y1)

            cv2.circle(image, (x1, y1), 10, (0, 0, 0), cv2.FILLED)

            for key, key_x, key_y, key_w, key_h in key_boxes:
                if key_x < x1 < key_x + key_w and key_y < y1 < key_y + key_h:
                    cv2.rectangle(image, (key_x, key_y), (key_x + key_w, key_y + key_h), (0, 0, 0), cv2.FILLED)
                    cv2.putText(image, key, (key_x + 10, key_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return left_hand_position


def isKeyPressed(leftHandPos, hoveredKeyPos, threshold=20):
    x1, y1 = leftHandPos
    xKey, yKey, width, height = hoveredKeyPos

    if (xKey <= x1 <= xKey + width and
        yKey <= y1 <= yKey + height):
        return True
    return False




def displayPressedKeys(image):
    y_offset = 600

    current_line = []
    for key in pressed_keys:
        if key == "\n":

            text = " ".join(current_line)
            cv2.putText(image, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_offset += 40
            current_line = []
        else:
            current_line.append(key)


    if current_line:
        text = " ".join(current_line)
        cv2.putText(image, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)




def handleKeyPress(key):
    global pressed_keys
    if key == "Backspace" and pressed_keys:
        pressed_keys.pop()  # Remove last key
    elif key == "Space":
        pressed_keys.append(" ")  # Add space
    elif key == "Enter":
        pressed_keys.append("\n")  # New line
    else:
        pressed_keys.append(key)  # Add normal character


def main():
    while True:
        ret, image = capture.read()
        if not ret:
            continue

        image, hands = detect.multiHandFinder(image)
        multiHandsData = detect.multiHandPositionFinder(image)

        findHands(image, hands)

        # Call function to draw keyboard keys on webcam
        key_boxes = draw_keyboard(image)

        # Check left hand index at a particular rectangle
        leftHandPos = leftHand(multiHandsData, image, key_boxes)

        # Check right hand thumb and index
        rightHand(multiHandsData, image, key_boxes, leftHandPos)

        # Show what has been typed
        displayPressedKeys(image)

        cv2.imshow("Virtual Keyboard", image)
        if cv2.waitKey(1) == ord("q"):
            break

    stringOutputter = "".join(pressed_keys)

    capture.release()
    cv2.destroyAllWindows()
    print("List format : \n",pressed_keys)
    print("String format : \n",stringOutputter)


if __name__ == "__main__":
    main()
