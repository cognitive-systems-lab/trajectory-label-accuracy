import numpy as np

def tlacc(frames_predicted, frames_target, direction_thresh = 5.0):
    """Evaluates F0 against a reference using the directory label accuracy 
    measure.

    The direction-label accuracy is accuracy with four labels: unvoiced, 
    voiced-flat, voiced-rising, voiced-falling. A frame is considered non-flat 
    if the difference between the previous and next frame is greater than 5Hz,
    extend-padded at borders of utterances and borders of unvoiced sections.

    This code was written with a frame shift of 10ms in mind, different frame
    shifts may require resampling or changing the threshold.

    Parameters:
        frames_predicted - the F0 frames to evaluate, in Hz (non-logarithmic)
        frames_target - the reference F0 frames, in Hz (non-logarithmic)
        direction_thresh - the threshold in Hz to consider a trajectory non-flat

    Returns the trajectory-label accuracy and the labels for both the predicted
    as well as the target frames.
    """
    
    # Arrays for storing comparison data
    labels_voicing_predicted = []
    labels_voicing_target = []
    labels_direction_predicted = []
    labels_direction_target = []
    
    # Extract voicing labels and valid frames
    for frame, (predicted, target) in enumerate(zip(frames_predicted, frames_target)):
        # Voicing
        if predicted == 0.0:
            labels_voicing_predicted.append('U')
        else:
            labels_voicing_predicted.append('V')
            
        if target == 0.0:
            labels_voicing_target.append('U')
        else:
            labels_voicing_target.append('V')
    
    # Extract trajectory labels
    for frame, (predicted, target) in enumerate(zip(frames_predicted, frames_target)):
        # Edge-aware gradient for trajectory labels
        left_val_predicted = frames_predicted[frame]
        left_val_target = frames_target[frame]
        if frame != 0:
            if labels_voicing_predicted[frame - 1] == 'V':
                left_val_predicted = frames_predicted[frame - 1]
            if labels_voicing_target[frame - 1] == 'V':
                left_val_target = frames_target[frame - 1]
            
        right_val_predicted = frames_predicted[frame]
        right_val_target = frames_target[frame]
        if frame != min_len - 1:
            if labels_voicing_predicted[frame + 1] == 'V':
                right_val_predicted = frames_predicted[frame + 1]
            if labels_voicing_target[frame + 1] == 'V':
                right_val_target = frames_target[frame + 1]
        
        direction_gradient_predicted = right_val_predicted - left_val_predicted
        direction_gradient_target = right_val_target - left_val_target
        
        # Direction labels
        if labels_voicing_predicted[frame] == 'V':
            if direction_gradient_predicted > direction_thresh:
                labels_direction_predicted.append('voiced-rising')
            elif direction_gradient_predicted < -direction_thresh:
                labels_direction_predicted.append('voiced-falling')
            else:
                labels_direction_predicted.append('voiced-flat')
        else:
            labels_direction_predicted.append('unvoiced')
            
        if labels_voicing_target[frame] == 'V':
            if direction_gradient_target > direction_thresh:
                labels_direction_target.append('voiced-rising')
            elif direction_gradient_target < -direction_thresh:
                labels_direction_target.append('voiced-falling')
            else:
                labels_direction_target.append('voiced-flat')
        else:
            labels_direction_target.append('unvoiced')
    
    # Calculate TLAcc
    direction_correct = len(np.where(np.array(labels_direction_predicted) == np.array(labels_direction_target))[0])
    direction_correct /= min_len
    
    return(direction_correct, labels_direction_predicted, labels_direction_target)
