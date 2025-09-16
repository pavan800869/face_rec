import sys
import cv2
import time
import numpy as np

# Add system packages path for picamera2
sys.path.append('/usr/lib/python3/dist-packages')

try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 not found. Please install with:")
    print("sudo apt install python3-picamera2")
    sys.exit(1)

def main():
    # Initialize camera
    picam2 = Picamera2()
    
    try:
        # Configure camera with proper settings for display
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)},
            display="main"
        )
        picam2.configure(config)
        
        # Start camera
        picam2.start()
        
        # Allow camera to warm up
        time.sleep(2)
        
        print("Camera started successfully!")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        print("- Press 'r' to toggle recording (if needed)")
        
        frame_count = 0
        
        while True:
            try:
                # Capture frame
                frame = picam2.capture_array()
                
                # Convert RGB to BGR for OpenCV display
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add frame counter and timestamp
                frame_count += 1
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame_bgr, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_bgr, timestamp, (10, 460), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow("PiCamera2 Test", frame_bgr)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"captured_frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame_bgr)
                    print(f"Frame saved as {filename}")
                elif key == ord('r'):
                    print("Recording toggle (not implemented)")
                    
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"Error capturing frame: {e}")
                break
                
    except Exception as e:
        print(f"Error initializing camera: {e}")
        print("Make sure:")
        print("1. Camera is enabled: sudo raspi-config > Interface Options > Camera")
        print("2. picamera2 is installed: sudo apt install python3-picamera2")
        print("3. You have proper permissions")
        
    finally:
        # Cleanup
        try:
            picam2.stop()
            print("Camera stopped")
        except:
            pass
        cv2.destroyAllWindows()
        print("Windows closed")

if __name__ == "__main__":
    main()