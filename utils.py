import math

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two (x, y) coordinates.
    """
    if point1 is None or point2 is None:
        return 0.0
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])


def get_player_with_ball(player_positions, ball_position, threshold=150):
    """
    Determine which player is closest to the ball within a distance threshold.
    
    Args:
        player_positions (dict): Map of player track_id -> (center_x, center_y)
        ball_position (tuple): (x, y) coordinates of the ball
        threshold (float): Maximum pixel distance to consider possession (default: 150)
        
    Returns:
        int/str or None: track_id of the player in possession, or None
    """
    if not player_positions or ball_position is None:
        return None

    closest_player = None
    min_dist = float('inf')

    for track_id, pos in player_positions.items():
        dist = calculate_distance(pos, ball_position)
        if dist < min_dist:
            min_dist = dist
            closest_player = track_id

    if min_dist <= threshold:
        return closest_player

    return None
