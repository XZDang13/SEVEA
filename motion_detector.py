import my_utils

motions = {
    "push": ["unwanted action for pick place.", 'reach puck.', "grasp puck.", 'move puck to sphere goal.', 'puck is near to sphere goal.'],
    "pushback": ["unwanted action for push back.", 'reach cube.', "grasp cube.", 'move cube to sphere goal.', 'cube is near to sphere goal.'],
    "draweropen": ["unwanted action for drawer open.", "reach drawer handle.", "hook drawer handler", "pull drawer inward.", "drawer is opened."],
    "buttonpress": ["unwanted action for button press.", "reach button.", "push button outward.", "button is pressed."],
    "windowopen": ["unwanted action for window open.", "reach window handle.", "slide window to right.", "window is opened."],
}

class BUTTONPRESS:
    motions = ["unwanted action for button press.", "reach button.", "push button outward.", "button is pressed."]
    @staticmethod
    def get_motion_label(obs, next_obs, is_grasped):
        xpos = my_utils.get_xpos(obs)
        next_xpos = my_utils.get_xpos(next_obs)

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "target", "goal", 0.024)

        is_target_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "target", 0.001)
        is_target_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "target", "goal", 0.6)

        is_gripper_moving_to_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "target", 0.6)
        is_gripper_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "gripper", 0.001)

        if is_goal_reached:
            return "button is pressed."

        if is_target_moving and is_target_moving_to_goal:
            return "push button outward."
        
        if is_gripper_moving and is_gripper_moving_to_target:
            return "reach button."
                
        return "unwanted action for button press."
            
class DRAWEROPEN:
    motions = ["unwanted action for drawer open.", "reach drawer handle.", "hook drawer handler", "pull drawer inward.", "drawer is opened."]
    @staticmethod
    def get_motion_label(obs, next_obs, is_grasped):
        
        xpos = my_utils.get_xpos(obs)
        next_xpos = my_utils.get_xpos(next_obs)

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "target", "goal", 0.03)
        
        is_target_reached = my_utils.DetectMotion.is_reached(xpos, "gripper", "target", 0.05)
        is_above_target = my_utils.DetectMotion.is_above(xpos, "gripper", "target", offset=0.05, aligned=False)
        is_above_target_aligned = my_utils.DetectMotion.is_above(xpos, "gripper", "target", aligned=True)

        is_target_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "target", 0.001)
        is_target_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "target", "goal", 0.5)
        
        is_gripper_moving_to_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "target", 0.5)
        is_gripper_moving_to_above_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "target", 0.6, offset=[0, 0, 0.05])
        is_gripper_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "gripper", 0.001)
        is_grasping = my_utils.DetectMotion.is_grasping(xpos, next_xpos)

        if is_goal_reached:
            return "drawer is opened."

        if is_target_reached and is_target_moving and is_target_moving_to_goal:
            return "pull drawer inward."

        if is_above_target_aligned and is_gripper_moving and is_gripper_moving_to_target:
            return "hook drawer handler"

        if is_gripper_moving and is_gripper_moving_to_above_target and is_above_target and not is_grasping:
            return "reach drawer handle."
                
        return "unwanted action for drawer open."
    
class PUSH:
    motions = ["unwanted action for pick place.", "reach puck.", "grasp puck.", "move puck to sphere goal."]#, "puck is near to sphere goal."]
    @staticmethod
    def get_motion_label(obs, next_obs, is_grasped):

        xpos = my_utils.get_xpos(obs)
        next_xpos = my_utils.get_xpos(next_obs)

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "target", "goal", 0.05)
        is_target_reached = my_utils.DetectMotion.is_reached(xpos, "gripper", "target", 0.04)

        is_target_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "target", 0.001)
        is_target_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "target", "goal", 0.65)

        is_gripper_moving_to_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "target", 0.65)
        is_gripper_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "gripper", 0.001)
        is_grasping = my_utils.DetectMotion.is_grasping(xpos, next_xpos)
        
        if is_goal_reached:
            return "puck is near to sphere goal."

        if (is_grasped and (is_target_moving and is_target_moving_to_goal)):
            return "move puck to sphere goal."

        if (is_target_reached and is_grasping) or (is_grasped and not is_target_moving):
            return "grasp puck."

        if ((is_gripper_moving and is_gripper_moving_to_target) or is_target_reached) and not is_target_moving and not is_grasping:
            return "reach puck."

        return "unwanted action for pick place."
    
class PUSHBACK:
    motions = ["unwanted action for push back.", "reach cube.", "grasp cube.", "move cube to sphere goal.", "cube is near to sphere goal."]
    @staticmethod
    def get_motion_label(obs, next_obs, is_grasped):

        xpos = my_utils.get_xpos(obs)
        next_xpos = my_utils.get_xpos(next_obs)

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "target", "goal", 0.07)
        is_target_reached = my_utils.DetectMotion.is_reached(xpos, "gripper", "target", 0.065)

        is_target_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "target", 0.0025)
        is_target_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "target", "goal", 0.5)

        is_gripper_moving_to_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "target", 0.6, offset=[0., 0., 0.05])
        is_gripper_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "gripper", 0.0025)
        is_grasping = my_utils.DetectMotion.is_grasping(xpos, next_xpos)
        
        if is_goal_reached:
            return "cube is near to sphere goal."

        if is_grasped and (is_target_moving and is_target_moving_to_goal):
            return "move cube to sphere goal."

        if (is_target_reached and is_grasping):
            return "grasp cube."

        if (is_gripper_moving and is_gripper_moving_to_target) and not is_target_moving and not is_grasping:
            return "reach cube." 

        return "unwanted action for push back."
        
class WINDOWOPEN:
    motions = ["unwanted action for window open.", "reach window handle.", "slide window to right.", "window is opened."]
    @staticmethod
    def get_motion_label(obs, next_obs, is_grasped):
        xpos = my_utils.get_xpos(obs)
        next_xpos = my_utils.get_xpos(next_obs)

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "target", "goal", 0.105)
        
        is_target_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "target", 0.001)
        is_target_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "target", "goal", 0.5)

        is_gripper_moving_to_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "target", 0.5)
        is_gripper_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "gripper", 0.001)

        if is_goal_reached:
            return "window is opened."

        if is_target_moving and is_target_moving_to_goal:
                return "slide window to right."

        if is_gripper_moving and is_gripper_moving_to_target:
            return "reach window handle."
                
        return "unwanted action for window open."
    
def get_motion_detecor(task):
 
    if task == "buttonpress":
        return BUTTONPRESS
    
    if task == "draweropen":
        return DRAWEROPEN
    
    if task == "push":
        return PUSH
    
    if task == "windowopen":
        return WINDOWOPEN
    
    if task == "pushback":
        return PUSHBACK
    
    return None
        