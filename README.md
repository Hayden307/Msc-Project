# Msc-Project
import cv2  # Import OpenCV library
import numpy as np  # Import numpy library for numerical operations

# Define color ranges for each Rubik's Cube color in HSV color space
color_ranges = {
    'red':    ([0, 50, 50],   [10, 255, 255]),    # Lower and upper bounds for red
    'orange': ([11, 50, 50],  [25, 255, 255]),    # Lower and upper bounds for orange
    'yellow': ([26, 50, 50],  [34, 255, 255]),    # Lower and upper bounds for yellow
    'green':  ([35, 50, 50],  [85, 255, 255]),    # Lower and upper bounds for green
    'blue':   ([86, 50, 50],  [125, 255, 255]),   # Lower and upper bounds for blue
    'white':  ([0, 0, 180],   [255, 30, 255]),    # Lower and upper bounds for white
}

# Function to find the centroid of a specific color in a frame
def find_color(frame, color_name):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert frame to HSV color space
    lower_bound, upper_bound = color_ranges[color_name]  # Get lower and upper bounds for the specified color
    lower_bound = np.array(lower_bound, dtype=np.uint8)  # Convert lower bound to numpy array
    upper_bound = np.array(upper_bound, dtype=np.uint8)  # Convert upper bound to numpy array
    mask = cv2.inRange(hsv, lower_bound, upper_bound)  # Create a mask using the specified color range
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the mask
    if contours:  # If contours are found
        contour = max(contours, key=cv2.contourArea)  # Get the largest contour
        M = cv2.moments(contour)  # Calculate moments of the contour
        if M["m00"] != 0:  # If the area of the contour is not zero
            cx = int(M["m10"] / M["m00"])  # Calculate centroid x-coordinate
            cy = int(M["m01"] / M["m00"])  # Calculate centroid y-coordinate
            return cx, cy  # Return centroid coordinates
    return None  # If no centroid found, return None

# Main function to capture video from webcam and detect colors
def main():
    cap = cv2.VideoCapture(0)  # Initialize webcam (0 is usually the default webcam)
    while True:  # Infinite loop to continuously process frames
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:  # If frame capture fails
            print("Failed to capture frame from webcam")
            break  # Break out of the loop
        
        # Loop through each color in the color ranges dictionary
        for color_name in color_ranges:
            center = find_color(frame, color_name)  # Find centroid of the specified color in the frame
            if center:  # If centroid is found
                # Draw text showing the color name at the centroid position on the frame
                cv2.putText(frame, color_name, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Rubik\'s Cube Color Detection', frame)  # Display the frame with color detection
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # If 'q' key is pressed
            break  # Break out of the loop

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()  # Call the main function when the script is executed
