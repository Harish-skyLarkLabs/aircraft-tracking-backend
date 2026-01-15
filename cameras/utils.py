"""
Utility functions for cameras app
"""
import cv2
import logging

logger = logging.getLogger(__name__)


def check_rtsp(rtsp_link: str, timeout: int = 10):
    """
    Check if an RTSP stream is accessible and return a thumbnail frame.
    
    Args:
        rtsp_link: The RTSP URL to check
        timeout: Connection timeout in seconds
        
    Returns:
        numpy.ndarray: The captured frame if successful, None otherwise
    """
    if not rtsp_link:
        return None

    try:
        cap = cv2.VideoCapture(rtsp_link)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            logger.warning(f"Failed to open RTSP stream: {rtsp_link}")
            return None

        # Try to read a frame
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            return frame
        else:
            logger.warning(f"Failed to read frame from RTSP stream: {rtsp_link}")
            return None

    except Exception as e:
        logger.error(f"Error checking RTSP stream {rtsp_link}: {str(e)}")
        return None


def get_stream_info(rtsp_link: str):
    """
    Get information about an RTSP stream.
    
    Args:
        rtsp_link: The RTSP URL to check
        
    Returns:
        dict: Stream information including resolution and FPS
    """
    if not rtsp_link:
        return None

    try:
        cap = cv2.VideoCapture(rtsp_link)
        
        if not cap.isOpened():
            return None

        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        
        cap.release()
        return info

    except Exception as e:
        logger.error(f"Error getting stream info for {rtsp_link}: {str(e)}")
        return None
