import json
import numpy as np

leap_data = open("/home/jaka/CollaborICE/LEAP/20250217-153506_leap.json")

def get_hand_pos():
    hand_pos_history = []
    hand_radius_history = []
    for l in leap_data:
        data = json.loads(l)
        hands = data.get('hands')
        hand_radius = 0
        hand_pos = np.zeros(3)
        if hands:
            # Filter out low-confidence hand detections
            confidences = np.array([h.get('confidence') for h in hands])
            hand = hands[np.argmax(confidences)]
            if hand.get('confidence') > 0.9:
                keypoints = hand.get('hand_keypoints')

                fingers = keypoints.get("fingers", {})
                palm_position = np.array(keypoints.get('palm_position'))

                joint_positions = np.array([joint_pos.get('prev_joint') for _, finger in fingers.items() 
                                                                        for _, joint_pos in finger.items()])
                if len(joint_positions) > 0:
                    # Compute a bounding sphere from the palm to the farthest away hand joint
                    dists = np.linalg.norm(joint_positions - palm_position, axis=1)
                    # Convert to JAKA world
                    hand_radius = np.max(dists) / 1000
                    R = np.array([
                        [0,-1,0],
                        [0,0,1],
                        [-1,0,0]])
                    t = np.array([0.400, 0, 0.025])
                    position = palm_position[:3]
                    hand_pos = (((np.array(position)) @ R) - t) / 1000
        hand_pos_history.append(hand_pos)
        hand_radius_history.append(hand_radius)
    return hand_pos_history, hand_radius_history

hand_pos = np.load('./hand_pos.npy')
hand_radius = np.load('./hand_radius.npy')

print(hand_pos.shape)
print(hand_radius.shape)

import matplotlib.pyplot as plt

plt.plot(hand_pos[:, 0])
plt.show()