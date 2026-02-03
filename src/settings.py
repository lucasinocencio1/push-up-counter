# All config in one place

# Angle thresholds (degrees) for elbow states
ANGLE_UP_THRESHOLD = 160      # "UP" (arm extended)
ANGLE_DOWN_THRESHOLD = 70     # "DOWN" (arm bent)

# Counter robustness
MIN_FRAMES_IN_STATE = 3       # debouncing (min frames per state)
SMOOTHING_WINDOW = 5          # moving average for angle

# Anti-cheat: hip must move a bit too
USE_HIP_CHECK = True
HIP_MIN_DELTA = 0.03          # fraction of frame height

# Sets
DEFAULT_TARGET_REPS = 12      # target reps per set
INACTIVITY_SECONDS = 5.0      # auto-pause when idle

# Video
CAM_INDEX = 0                 # webcam index
DRAW_SKELETON = True          # draw landmarks

# Audio
BEEP_ON_REP = True            # beep on each rep
BEEP_ON_SET = True            # beep when set is done
