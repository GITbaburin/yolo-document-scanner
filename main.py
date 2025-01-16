from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np
import re
from datetime import datetime
from collections import defaultdict

def validate_dob(dob_str):
    try:
        dob = datetime.strptime(dob_str, "%Y/%m/%d")
    except ValueError:
        return False

    today = datetime.today()
    if dob > today:
        return False

    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    if age < 18 or age > 100:
        return False

    return True

def levenshtein_distance(s1, s2):
    """Compute the Levenshtein distance between two strings."""
    dp = [[0]*(len(s2)+1) for _ in range(len(s1)+1)]
    for i in range(len(s1)+1):
        dp[i][0] = i
    for j in range(len(s2)+1):
        dp[0][j] = j
    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # Deletion
                dp[i][j-1] + 1,      # Insertion
                dp[i-1][j-1] + cost  # Replacement
            )
    return dp[len(s1)][len(s2)]

def is_close_match(candidate, target):
    """
    Check if candidate matches target ignoring case.
    Accept if:
    - Exact case-insensitive match, OR
    - Levenshtein distance <= 2
    """
    candidate = candidate.lower().strip()
    target = target.lower().strip()
    if candidate == target:
        return True
    dist = levenshtein_distance(candidate, target)
    return dist <= 2

def clean_text(t):
    # Convert to lowercase
    t = t.lower()
    # Keep only letters, digits, slash, and spaces
    t = re.sub(r'[^a-z0-9/ ]+', '', t)
    # Collapse multiple spaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_information(clean_texts):
    extracted_info = {
        'surname': '',
        'given_name': '',
        'sex': '',
        'date_of_birth': ''
    }

    # Surname
    for i, t in enumerate(clean_texts):
        if is_close_match(t, 'surname') and i + 1 < len(clean_texts):
            extracted_info['surname'] = clean_texts[i + 1].strip().upper()

    # Given Name
    for i, t in enumerate(clean_texts):
        if is_close_match(t, 'given name') and i + 1 < len(clean_texts):
            extracted_info['given_name'] = clean_texts[i + 1].strip().upper()

    # Sex: just check if 'sex' is in the line, no fuzzy match needed here
    for t in clean_texts:
        # We know 'sex' is a short keyword, and OCR is fairly good at short words
        # If you still want fuzzy logic here, you can do a substring check or
        # simplify to just: if 'sex' in t
        if 'sex' in t:
            match = re.search(r'sex\s*(m|f)', t)
            if match:
                extracted_info['sex'] = match.group(1).upper()
                break  # Found sex once, stop searching

    # Date of Birth
    dob_pattern = r'(\d{4}/\d{2}/\d{2})'
    for t in clean_texts:
        match = re.search(dob_pattern, t)
        if match and not extracted_info['date_of_birth']:
            dob_candidate = match.group(1)
            if validate_dob(dob_candidate):
                extracted_info['date_of_birth'] = dob_candidate.upper()

    return extracted_info

def get_final_value(counts_dict):
    if not counts_dict:
        return ''
    return max(counts_dict, key=counts_dict.get)

def process_video(video_path, output_path):
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Choose a slower FPS, for example half of the original
    slower_fps = original_fps / 2.0

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          slower_fps,
                          (width + 300, height))

    surname_counts = defaultdict(int)
    givenname_counts = defaultdict(int)
    sex_counts = defaultdict(int)
    dob_counts = defaultdict(int)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Process every 5 frames
        if frame_count % 5 != 0:
            continue

        print(f'Processing frame {frame_count}/{total_frames}')

        # YOLO detection
        results = model(frame)

        text_panel = np.ones((height, 300, 3), dtype=np.uint8) * 255

        found_box = False
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Only proceed if the object is a "book"
                if class_name.lower() == 'book':
                    found_box = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, width), min(y2, height)

                    # Compute 10% margin
                    box_w = x2 - x1
                    box_h = y2 - y1
                    margin_x = int(box_w * 0.1)
                    margin_y = int(box_h * 0.1)

                    # Expanded coordinates for OCR
                    ocr_x1 = max(x1 - margin_x, 0)
                    ocr_y1 = max(y1 - margin_y, 0)
                    ocr_x2 = min(x2 + margin_x, width)
                    ocr_y2 = min(y2 + margin_y, height)

                    # Extract the region for OCR
                    doc_region = frame[ocr_y1:ocr_y2, ocr_x1:ocr_x2]
                    doc_region = cv2.resize(doc_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                    # OCR on the unmodified image
                    ocr_text = pytesseract.image_to_string(doc_region)
                    raw_texts = [line.strip() for line in ocr_text.split('\n') if line.strip()]
                    clean_texts = [clean_text(line) for line in raw_texts if line]

                    print("OCR texts from detected box (raw):", raw_texts)
                    print("OCR texts from detected box (cleaned):", clean_texts)

                    current_info = extract_information(clean_texts)
                    if current_info['surname']:
                        surname_counts[current_info['surname']] += 1
                    if current_info['given_name']:
                        givenname_counts[current_info['given_name']] += 1
                    if current_info['sex']:
                        sex_counts[current_info['sex']] += 1
                    if current_info['date_of_birth']:
                        dob_counts[current_info['date_of_birth']] += 1

                    # Draw rectangle only around the original box (no margin)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if not found_box:
            full_ocr = pytesseract.image_to_string(frame)
            full_raw_texts = [line.strip() for line in full_ocr.split('\n') if line.strip()]
            full_clean_texts = [clean_text(line) for line in full_raw_texts if line]
            print("No boxes found. Full frame OCR texts (raw):", full_raw_texts)
            print("No boxes found. Full frame OCR texts (cleaned):", full_clean_texts)

        # Temporary best guess
        temp_surname = get_final_value(surname_counts)
        temp_given_name = get_final_value(givenname_counts)
        temp_sex = get_final_value(sex_counts)
        temp_dob = get_final_value(dob_counts)

        y_text = 30
        cv2.putText(text_panel, "Extracted Information:", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_text += 40

        cv2.putText(text_panel, f"Surname: {temp_surname}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_text += 30

        cv2.putText(text_panel, f"Given Name: {temp_given_name}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_text += 30

        cv2.putText(text_panel, f"Sex: {temp_sex}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_text += 30

        cv2.putText(text_panel, f"Date of Birth: {temp_dob}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        combined_frame = np.hstack((frame, text_panel))
        out.write(combined_frame)

    cap.release()

    final_surname = get_final_value(surname_counts)
    final_given_name = get_final_value(givenname_counts)
    final_sex = get_final_value(sex_counts)
    final_dob = get_final_value(dob_counts)

    print("\nFinal Extracted Information:")
    print(f"Surname: {final_surname}")
    print(f"Given Name: {final_given_name}")
    print(f"Sex: {final_sex}")
    print(f"Date of Birth: {final_dob}")

    final_info_panel = np.ones((height, 300, 3), dtype=np.uint8) * 255
    y_text = 30
    cv2.putText(final_info_panel, "Final Extracted Information:", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y_text += 40

    cv2.putText(final_info_panel, f"Surname: {final_surname}", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    y_text += 30
    cv2.putText(final_info_panel, f"Given Name: {final_given_name}", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    y_text += 30
    cv2.putText(final_info_panel, f"Sex: {final_sex}", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    y_text += 30
    cv2.putText(final_info_panel, f"Date of Birth: {final_dob}", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    final_frame = np.zeros((height, width, 3), dtype=np.uint8)
    final_combined = np.hstack((final_frame, final_info_panel))
    frames_to_write = int(5 * slower_fps)
    for _ in range(frames_to_write):
        out.write(final_combined)
    out.release()

    return {
        'surname': final_surname,
        'given_name': final_given_name,
        'sex': final_sex,
        'date_of_birth': final_dob
    }

# Example usage:
video_path = 'test_for_yolo.mp4'
output_path = 'processed_output.mp4'
extracted_info = process_video(video_path, output_path)
