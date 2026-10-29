from ultralytics import YOLO
import cv2


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolo11n.pt"
CONFIDENCE = 0.50
CAMERA_ID = 0


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Camera open nahi ho raha.")
    print("Camera ID change karke try karo:")
    print("CAMERA_ID = 1")
    exit()


# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = "YOLO Object Detection + Spectacles"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)


# ============================================================
# START
# ============================================================

print()
print("==============================================")
print("     YOLO OBJECT DETECTION SYSTEM")
print("==============================================")
print("Camera Started")
print()
print("Objects will be detected automatically.")
print("Press Q to Exit")
print("==============================================")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ CAMERA
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Camera frame read nahi ho raha.")

        break


    # --------------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )


    # --------------------------------------------------------
    # OBJECT COUNT
    # --------------------------------------------------------

    object_count = {}


    # --------------------------------------------------------
    # PROCESS RESULTS
    # --------------------------------------------------------

    for result in results:

        boxes = result.boxes


        if boxes is None:
            continue


        # ----------------------------------------------------
        # LOOP THROUGH DETECTED OBJECTS
        # ----------------------------------------------------

        for box in boxes:

            # Coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Confidence
            confidence = float(
                box.conf[0]
            )


            # Class ID
            class_id = int(
                box.cls[0]
            )


            # Object name
            object_name = model.names[
                class_id
            ]


            # ------------------------------------------------
            # COUNT OBJECT
            # ------------------------------------------------

            if object_name not in object_count:

                object_count[object_name] = 0


            object_count[object_name] += 1


            # ------------------------------------------------
            # DRAW OBJECT BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # OBJECT LABEL
            # ------------------------------------------------

            label = (
                f"{object_name} "
                f"{confidence * 100:.1f}%"
            )


            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(y1 - 10, 25)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 0),
                2
            )


    # ========================================================
    # PERSON DETECTION
    # ========================================================

    person_count = object_count.get(
        "person",
        0
    )


    # ========================================================
    # SPECTACLES INFORMATION
    # ========================================================
    #
    # Standard YOLO11 model does NOT contain
    # "spectacles" as a class.
    #
    # Therefore we do NOT falsely label objects
    # as spectacles here.
    #
    # If a custom spectacles model is available,
    # it can be loaded separately.
    # ========================================================

    spectacles_text = "Spectacles: Custom Model Required"


    # ========================================================
    # TOP INFORMATION BAR
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 55),
        (25, 25, 25),
        -1
    )


    cv2.putText(
        frame,
        "YOLO OBJECT DETECTION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (255, 255, 255),
        2
    )


    # ========================================================
    # PERSON COUNT
    # ========================================================

    cv2.putText(
        frame,
        f"Persons: {person_count}",
        (400, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # ========================================================
    # OBJECT COUNT PANEL
    # ========================================================

    panel_x = 10
    panel_y = 70

    panel_width = 280

    panel_height = (
        45 +
        len(object_count) * 28
    )


    cv2.rectangle(
        frame,
        (
            panel_x,
            panel_y
        ),
        (
            panel_x + panel_width,
            panel_y + panel_height
        ),
        (30, 30, 30),
        -1
    )


    # Panel title

    cv2.putText(
        frame,
        "OBJECT COUNTS",
        (
            panel_x + 10,
            panel_y + 30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY COUNTS
    # ========================================================

    count_y = panel_y + 60


    for name, count in object_count.items():

        count_text = (
            f"{name}: {count}"
        )


        cv2.putText(
            frame,
            count_text,
            (
                panel_x + 10,
                count_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        count_y += 28


    # ========================================================
    # SPECTACLES STATUS
    # ========================================================

    cv2.rectangle(
        frame,
        (
            10,
            frame.shape[0] - 85
        ),
        (
            430,
            frame.shape[0] - 50
        ),
        (30, 30, 30),
        -1
    )


    cv2.putText(
        frame,
        spectacles_text,
        (
            20,
            frame.shape[0] - 62
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255),
        1
    )


    # ========================================================
    # EXIT TEXT
    # ========================================================

    cv2.putText(
        frame,
        "Press Q to Exit",
        (
            20,
            frame.shape[0] - 15
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        print("Exiting...")

        break


# ============================================================
# RELEASE CAMERA
# ============================================================

cap.release()


# ============================================================
# CLOSE WINDOWS
# ============================================================

cv2.destroyAllWindows()


print("Camera closed.")
print("Program finished.")