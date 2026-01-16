#!/usr/bin/env python3
"""
Quick test script to verify LiveKit streaming works.
This publishes a test video pattern to LiveKit.
"""
import asyncio
import numpy as np
import cv2
import time
from livekit import rtc, api

# Configuration
LIVEKIT_URL = "ws://localhost:7880"
API_KEY = "devkey"
API_SECRET = "secret"
ROOM_NAME = "test-room"
PARTICIPANT_NAME = "test-publisher"


async def main():
    print(f"Connecting to LiveKit at {LIVEKIT_URL}...")
    
    # Create access token with new API
    token = (
        api.AccessToken(api_key=API_KEY, api_secret=API_SECRET)
        .with_identity(PARTICIPANT_NAME)
        .with_name(PARTICIPANT_NAME)
        .with_grants(api.VideoGrants(
            room_join=True,
            room=ROOM_NAME,
            can_publish=True,
            can_subscribe=True,
        ))
    )
    jwt_token = token.to_jwt()
    print(f"Token created for room: {ROOM_NAME}")
    
    # Create room and connect
    room = rtc.Room()
    
    @room.on("participant_connected")
    def on_participant_connected(participant):
        print(f"Participant connected: {participant.identity}")
    
    @room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        print(f"Participant disconnected: {participant.identity}")
    
    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        print(f"Track subscribed: {track.kind} from {participant.identity}")
    
    await room.connect(LIVEKIT_URL, jwt_token)
    print(f"Connected to room: {room.name}")
    
    # Create video source
    video_source = rtc.VideoSource(width=640, height=480)
    video_track = rtc.LocalVideoTrack.create_video_track("camera", video_source)
    
    # Publish track
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    publication = await room.local_participant.publish_track(video_track, options)
    print(f"Published video track: {publication.sid}")
    
    print("\n" + "="*50)
    print("LiveKit is working! Publishing test video...")
    print(f"Room: {ROOM_NAME}")
    print(f"View at: http://localhost:7880 (LiveKit dashboard)")
    print("="*50 + "\n")
    
    # Publish test frames
    frame_count = 0
    try:
        while True:
            # Create a test frame with moving pattern
            frame = np.zeros((480, 640, 4), dtype=np.uint8)
            
            # Add gradient background
            for y in range(480):
                frame[y, :, 0] = int(255 * y / 480)  # Blue gradient
                frame[y, :, 1] = int(128 * (1 - y / 480))  # Green
                frame[y, :, 2] = 50  # Red
                frame[y, :, 3] = 255  # Alpha
            
            # Add moving circle
            cx = int(320 + 200 * np.sin(frame_count * 0.05))
            cy = int(240 + 100 * np.cos(frame_count * 0.05))
            cv2.circle(frame, (cx, cy), 50, (255, 255, 255, 255), -1)
            
            # Add text
            text = f"LiveKit Test - Frame {frame_count}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2)
            cv2.putText(frame, f"Room: {ROOM_NAME}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200, 255), 1)
            cv2.putText(frame, "Connect from frontend to view", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150, 255), 1)
            
            # Create video frame
            video_frame = rtc.VideoFrame(
                width=640,
                height=480,
                type=rtc.VideoBufferType.RGBA,
                data=frame.tobytes(),
            )
            
            # Capture frame
            video_source.capture_frame(video_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Published {frame_count} frames...")
            
            await asyncio.sleep(1/30)  # 30 FPS
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await room.disconnect()
        print("Disconnected from LiveKit")


if __name__ == "__main__":
    asyncio.run(main())

