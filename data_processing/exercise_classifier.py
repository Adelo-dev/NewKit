def classify_exercise_from_angles(angles: dict) -> str:
    """
    Classifies the exercise based on joint angles.

    Args:
        angles (dict): Dictionary with joint angles in degrees.

    Returns:
        str: Detected exercise label.
    """
    elbow_avg = (angles["left_elbow"] + angles["right_elbow"]) / 2
    knee_avg = (angles["left_knee"] + angles["right_knee"]) / 2
    hip_avg = (angles["left_hip"] + angles["right_hip"]) / 2
    print(elbow_avg, knee_avg, hip_avg)
    if elbow_avg < 70 and knee_avg > 160 and hip_avg > 160:
        return "Pushup"
    elif knee_avg < 100 and hip_avg < 100:
        return "Squat"
    elif elbow_avg > 160 and knee_avg > 160 and hip_avg > 160:
        return "Plank"
    else:
        return "Unknown"
