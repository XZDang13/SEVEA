import my_utils


motions = {
    "hammer": ["unwanted action for drive nail to wall.", "reach hammer.", "grasp hammer.",  "move hammer to nail.", "knocking nail to wall.", "nail is drived to wall."],
    "pushback": ["unwanted action for push back.", 'reach cube.', "grasp cube.", 'move cube to sphere goal.', 'cube is near to sphere goal.'],
    "draweropen": ["unwanted action for drawer open.", "reach drawer handle.", "hook drawer handler", "pull drawer inward.", "drawer is opened."],
    "buttonpress": ["unwanted action for button press.", "reach button.", "push button outward.", "button is pressed."],
    "windowopen": ["unwanted action for window open.", "reach window handle.", "slide window to right.", "window is opened."],
    "pickplacewall": ["unwanted action for pick place with wall.", "reach puck.", "grasp puck.", "move puck cross wall.",  "move puck to goal.", "puck is near to goal."],
    "peginsert": ["unwanted action for peg insert.", "reach cube.", "grasp cube.",  "move cube to goal.", "cube is near to goal."],
    "reach": ["unwanted action for reach.", "reach goal.", "end effector is near to goal."],
    "pickplace": ["unwanted action for push back.", "reach cube.", "grasp cube.", "move cube to sphere goal.", "cube is near to sphere goal."],
    "soccer": ["unwanted action for soccer.", "reach soccer.", "grasp soccer.",  "move soccer to goal.", "soccer is near to goal."]
}

class BUTTONPRESS:
    motions = ["unwanted action for button press.", "reach button.", "push button outward.", "button is pressed."]
    @staticmethod
    def get_motion_label(obs, next_obs, is_grasped):
        xpos = my_utils.get_xpos(obs)
        next_xpos = my_utils.get_xpos(next_obs)

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "object_1", "goal", 0.024)

        is_target_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "object_1", 0.001)
        is_target_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "object_1", "goal", 0.6)

        is_gripper_moving_to_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "object_1", 0.6)
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

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "object_1", "goal", 0.03)
        
        is_target_reached = my_utils.DetectMotion.is_reached(xpos, "gripper", "object_1", 0.05)
        is_above_target = my_utils.DetectMotion.is_above(xpos, "gripper", "object_1", offset=0.05, aligned=False)
        is_above_target_aligned = my_utils.DetectMotion.is_above(xpos, "gripper", "object_1", aligned=True)

        is_target_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "object_1", 0.001)
        is_target_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "object_1", "goal", 0.5)
        
        is_gripper_moving_to_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "object_1", 0.5)
        is_gripper_moving_to_above_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "object_1", 0.6, offset=[0, 0, 0.05])
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
        
class PICKPLACWALL:
    motions = ["unwanted action for pick place with wall.", "reach puck.", "grasp puck.", "move puck cross wall.",  "move puck to goal.", "puck is near to goal."]
    @staticmethod
    def get_motion_label(obs, next_obs, is_grasped):

        xpos = my_utils.get_xpos(obs)
        next_xpos = my_utils.get_xpos(next_obs)

        xpos["wall"] =  [0.1, 0.75, .06]
        next_xpos["wall"] =  [0.1, 0.75, .06]

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "object_1", "goal", 0.07, offset=[0.0, 0.0, 0.0])
        is_object_1_reached = my_utils.DetectMotion.is_reached(xpos, "gripper", "object_1", 0.035)

        is_object_1_front_of_wall = my_utils.DetectMotion.is_front(xpos, "object_1", "wall", offset=-0.05)

        is_object_1_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "object_1", 0.001)
        is_object_1_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "object_1", "goal", 0.65, offset=[0.0, 0, 0.0])
        is_object_1_moving_cross_wall = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "object_1", "wall", 0.65, offset=[0.0, -0.025, 0.15])

        is_gripper_moving_to_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "object_1", 0.65)
        is_gripper_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "gripper", 0.001)
        is_grasping = my_utils.DetectMotion.is_grasping(xpos, next_xpos)
        
        if is_goal_reached:
            return "puck is near to goal."
        
        if (is_object_1_reached and not is_object_1_front_of_wall and (is_object_1_moving and is_object_1_moving_to_goal)):
            return "move puck to goal."
        
        if (is_object_1_reached and is_object_1_front_of_wall and (is_object_1_moving and is_object_1_moving_cross_wall)):
            return "move puck cross wall."

        if (is_object_1_reached and is_grasping):
            return "grasp puck."

        if ((is_gripper_moving and is_gripper_moving_to_target)) and not is_object_1_moving and not is_grasping:
            return "reach puck."

        return "unwanted action for pick place with wall."
    
class SOCCER:
    motions = ["unwanted action for soccer.", "reach soccer.", "grasp soccer.",  "move soccer to goal.", "soccer is near to goal."]
    @staticmethod
    def get_motion_label(obs, next_obs, is_grasped):

        xpos = my_utils.get_xpos(obs)
        next_xpos = my_utils.get_xpos(next_obs)

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "object_1", "goal", 0.05, offset=[0., 0, 0.0])
        is_object_1_reached = my_utils.DetectMotion.is_reached(xpos, "gripper", "object_1", 0.05)

        is_object_1_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "object_1", 0.001)
        is_object_1_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "object_1", "goal", 0.65, offset=[0., 0, 0.0])

        is_gripper_moving_to_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "object_1", 0.65)
        is_gripper_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "gripper", 0.001)
        is_grasping = my_utils.DetectMotion.is_grasping(xpos, next_xpos)
        
        if is_goal_reached:
            return "soccer is near to goal."
        
        if (is_object_1_reached and (is_object_1_moving and is_object_1_moving_to_goal)):
            return "move soccer to goal."

        if (is_object_1_reached and is_grasping):
            return "grasp soccer."

        if ((is_gripper_moving and is_gripper_moving_to_target)) and not is_object_1_moving and not is_grasping:
            return "reach soccer."

        return "unwanted action for soccer."
    
class PEGINSERT:
    motions = ["unwanted action for peg insert.", "reach cube.", "grasp cube.",  "move cube to goal.", "cube is near to goal."]
    @staticmethod
    def get_motion_label(obs, next_obs, is_grasped):

        xpos = my_utils.get_xpos(obs)
        next_xpos = my_utils.get_xpos(next_obs)

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "object_1", "goal", 0.04, offset=[0.17, 0, 0])
        is_object_1_reached = my_utils.DetectMotion.is_reached(xpos, "gripper", "object_1", 0.05)

        is_object_1_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "object_1", 0.001)
        is_object_1_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "object_1", "goal", 0.65, offset=[0.17, 0, 0])

        is_gripper_moving_to_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "object_1", 0.65)
        is_gripper_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "gripper", 0.001)
        is_grasping = my_utils.DetectMotion.is_grasping(xpos, next_xpos)
        
        if is_goal_reached:
            return "cube is near to goal."
        
        if (is_object_1_reached and (is_object_1_moving and is_object_1_moving_to_goal)):
            return "move cube to goal."

        if (is_object_1_reached and is_grasping):
            return "grasp cube."

        if ((is_gripper_moving and is_gripper_moving_to_target)) and not is_object_1_moving and not is_grasping:
            return "reach cube."

        return "unwanted action for peg insert."
    
class HAMMER:
    motions = ["unwanted action for drive nail to wall.", "reach hammer.", "grasp hammer.",  "move hammer to nail.", "knocking nail to wall.", "nail is drived to wall."]
    @staticmethod
    def get_motion_label(obs, next_obs, is_grasped):

        xpos = my_utils.get_xpos(obs)
        next_xpos = my_utils.get_xpos(next_obs)

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "object_2", "goal", 0.05)
        is_object_2_reached = my_utils.DetectMotion.is_reached(xpos, "object_1", "object_2", 0.065, offset=[-0.15, -0.05, 0.0])
        is_object_1_reached = my_utils.DetectMotion.is_reached(xpos, "gripper", "object_1", 0.05, offset=[-0.075, 0, 0])

        is_object_2_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "object_2", 0.001)
        is_object_2_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "object_2", "goal", 0.75)


        is_object_1_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "object_1", 0.001)
        is_object_1_moving_to_object_2 = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "object_1", "object_2", 0.75, offset=[-0.1, 0., 0.0])

        is_gripper_moving_to_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "object_1", 0.75)
        is_gripper_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "gripper", 0.001)
        is_grasping = my_utils.DetectMotion.is_grasping(xpos, next_xpos)
        
        if is_goal_reached and is_object_2_reached:
            return "nail is drived to wall."

        if (is_object_1_reached and is_object_2_reached and (is_object_2_moving and is_object_2_moving_to_goal)):
            return "knocking nail to wall."
        
        if (is_object_1_reached and (is_object_1_moving and is_object_1_moving_to_object_2)) and not is_object_2_moving:
            return "move hammer to nail."

        if (is_object_1_reached and is_grasping):
            return "grasp hammer."

        if ((is_gripper_moving and is_gripper_moving_to_target)) and not is_object_1_moving and not is_grasping:
            return "reach hammer."

        return "unwanted action for drive nail to wall."

class PICKPLACE:
    motions = ["unwanted action for push back.", "reach cube.", "grasp cube.", "move cube to sphere goal.", "cube is near to sphere goal."]
    @staticmethod
    def get_motion_label(obs, next_obs, is_grasped):

        xpos = my_utils.get_xpos(obs)
        next_xpos = my_utils.get_xpos(next_obs)

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "object_1", "goal", 0.05)
        is_target_reached = my_utils.DetectMotion.is_reached(xpos, "gripper", "object_1", 0.035)

        is_target_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "object_1", 0.001)
        is_target_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "object_1", "goal", 0.65)

        is_gripper_moving_to_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "object_1", 0.65)
        is_gripper_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "gripper", 0.001)
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

class PUSHBACK:
    motions = ["unwanted action for push back.", "reach cube.", "grasp cube.", "move cube to sphere goal.", "cube is near to sphere goal."]
    @staticmethod
    def get_motion_label(obs, next_obs, is_grasped):

        xpos = my_utils.get_xpos(obs)
        next_xpos = my_utils.get_xpos(next_obs)

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "object_1", "goal", 0.07)
        is_target_reached = my_utils.DetectMotion.is_reached(xpos, "gripper", "object_1", 0.065)

        is_target_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "object_1", 0.0025)
        is_target_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "object_1", "goal", 0.5)

        is_gripper_moving_to_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "object_1", 0.6, offset=[0., 0., 0.05])
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

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "object_1", "goal", 0.105)
        
        is_target_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "object_1", 0.001)
        is_target_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "object_1", "goal", 0.5)

        is_gripper_moving_to_target = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "object_1", 0.5)
        is_gripper_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "gripper", 0.001)

        if is_goal_reached:
            return "window is opened."

        if is_target_moving and is_target_moving_to_goal:
                return "slide window to right."

        if is_gripper_moving and is_gripper_moving_to_target:
            return "reach window handle."
                
        return "unwanted action for window open."
    
class REACH:
    motions = ["unwanted action for reach.", "reach goal.", "end effector is near to goal."]
    @staticmethod
    def get_motion_label(obs, next_obs, is_grasped):

        xpos = my_utils.get_xpos(obs)
        next_xpos = my_utils.get_xpos(next_obs)

        is_goal_reached = my_utils.DetectMotion.is_reached(xpos, "gripper", "goal", 0.03, offset=[0, 0, 0.05])
        

        is_gripper_moving_to_goal = my_utils.DetectMotion.is_moving_to(xpos, next_xpos, "gripper", "goal", 0.65, offset=[0, 0, 0.05])
        is_gripper_moving = my_utils.DetectMotion.is_moving(xpos, next_xpos, "gripper", 0.001)
        is_grasping = my_utils.DetectMotion.is_grasping(xpos, next_xpos)
        
        if is_goal_reached:
            return "end effector is near to goal."

        if ((is_gripper_moving and is_gripper_moving_to_goal)):
            return "reach goal."

        return "unwanted action for reach."
    
def get_motion_detecor(task):
 
    if task == "buttonpress":
        return BUTTONPRESS
    
    if task == "draweropen":
        return DRAWEROPEN
    
    if task == "pickplace":
        return PICKPLACE
    
    if task == "pickplacewall":
        return PICKPLACWALL
    
    if task == "hammer":
        return HAMMER

    if task == "soccer":
        return SOCCER
    
    if task == "peginsert":
        return PEGINSERT

    if task == "windowopen":
        return WINDOWOPEN
    
    if task == "pushback":
        return PUSHBACK
    
    if task == "reach":
        return REACH
    
    return None