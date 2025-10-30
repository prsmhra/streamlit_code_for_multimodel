from src.python.app.constants.constants import Constants


class DetectBlendshapeAction:

    def __init__(self):
        pass

    # removed from list - 38-mouthRight, 27-mouthDimpleLeft, 28-mouthDimpleRight
    # 29-mouthFrownLeft, 30-mouthFrownRight,
    def check_thresh(self,key):
        if key in [5,6, 7,22,23,25, 26, 50, 51]: # cheekPuff, cheekSquintLeft, cheekSquintRight, jawForward, jawLeft, jawRight, mouthClose,
                                    # mouthDimpleLeft, mouthDimpleRight, mouthFrownLeft, mouthFrownRight,mouthRight, noseSneerLeft, noseSneerRight, 
            check = self.diff_val > 0.33 * self.min_val_in_queue
            return check
        elif key in [27,28,29, 30,38]:
            check = self.diff_val/self.min_val_in_queue > 0.33 and self.diff_val>0.002
            return check
        else: 
            if self.diff_val/self.min_val_in_queue > 0.33 and self.diff_val>0.02 :
                return True
            else:
                return False
            
    def check_minima(self):
        if (self.uni_counter >=Constants.COUNT_ZERO or self.dec_counter>Constants.COUNT_ZERO)\
            and self.inc_counter==Constants.ONE and self.last_state==Constants.JUMP_INC_STATE:
            self.minima_idx = self.frame_count
            self.uni_counter = Constants.ZERO
            self.dec_counter = Constants.ZERO
        elif (self.uni_counter==Constants.COUNT_ONE and self.last_state==Constants.JUMP_UNI_STATE) and self.dec_counter>Constants.COUNT_ZERO\
            and self.inc_counter==Constants.COUNT_ZERO:
            self.minima_idx = self.frame_count
            self.uni_counter = Constants.ZERO
            self.dec_counter = Constants.ZERO

    def check_maxima(self):
        if (self.uni_counter >=Constants.COUNT_ZERO or self.inc_counter>Constants.COUNT_ZERO)\
            and self.dec_counter==Constants.ONE and self.last_state==Constants.JUMP_DEC_STATE:
            self.maxima_idx = self.frame_count
            self.uni_counter = Constants.ZERO
            self.inc_counter = Constants.ZERO
        elif self.inc_counter>Constants.COUNT_ZERO and (self.uni_counter==1 and self.last_state==Constants.JUMP_UNI_STATE)\
            and self.dec_counter==Constants.COUNT_ZERO:
            self.maxima_idx = self.frame_count
            self.uni_counter = Constants.ZERO
            self.inc_counter = Constants.ZERO

    def get_max_min_diff_vals(self):
        self.max_val_in_queue = max(self.running_avg_queue)
        self.max_val_index = self.running_avg_queue.index(max(self.running_avg_queue))
        self.min_val_in_queue = min(self.running_avg_queue)
        self.min_val_index = self.running_avg_queue.index(min(self.running_avg_queue))
        self.diff_val = self.max_val_in_queue - self.min_val_in_queue

    def identify_queue_state(self):
        if self.max_val_index >= Constants.COUNT_FIVE and self.min_val_index <= Constants.COUNT_TWO:
            self.inc_counter += Constants.COUNT_ONE
            self.last_state = Constants.JUMP_INC_STATE
        elif self.max_val_index <= Constants.COUNT_TWO and self.min_val_index >= Constants.COUNT_FIVE:
            self.dec_counter += Constants.COUNT_ONE
            self.last_state = Constants.JUMP_DEC_STATE
        else:
            self.uni_counter += Constants.COUNT_ONE
            self.last_state = Constants.JUMP_UNI_STATE

    def assign_queue_state(self):
        if self.maxima_idx==self.frame_count:
            self.queue_state = Constants.MAXIMA
        elif self.minima_idx==self.frame_count:
            self.queue_state = Constants.MINIMA
        elif self.last_state == Constants.JUMP_INC_STATE:
            self.queue_state = Constants.JUMP_INC_STATE
        elif self.last_state == Constants.JUMP_DEC_STATE:
            self.queue_state = Constants.JUMP_DEC_STATE
        elif self.last_state == Constants.JUMP_UNI_STATE:
            self.queue_state = Constants.JUMP_UNI_STATE


    def queue_mean(self,queue):
        return sum(queue)/len(queue)

    def get_action_state(self,key):


        # selecting thresholds based on keys
        if key in [5,6, 7,22,23,25, 26, 49,50]:
            diff_thresh = 0
            start_stop_val_thresh = 0.000001
        elif  key in [27,28,29, 30,33, 34, 38,51,52,53,54]:   # 33, 34 mouthLowerDownLeft, mouthLowerDownRight added
            diff_thresh = 0.0015
            start_stop_val_thresh = diff_thresh/1.25
        else:
            diff_thresh = 0.02
            start_stop_val_thresh = diff_thresh/1.25

        if key in [5]:
            max_thresh = 1.5
        elif key in [51,52,53,54]:
            max_thresh = 1.25
        else:
            max_thresh = 3
        


        if self.diff_val/self.min_val_in_queue>0.33 and self.diff_val>diff_thresh:

            if self.start_stop_val<start_stop_val_thresh and self.start_stop_flag==Constants.START:
                    self.start_stop_val = self.queue_mean(self.running_avg_queue)
            

            if self.last_state==Constants.JUMP_INC_STATE and self.start_stop_flag==Constants.STOP:
                self.start_stop_flag = Constants.START
                self.start_stop_val = self.queue_mean(self.running_avg_queue)
                

            elif self.minima_idx==self.frame_count and self.last_state==Constants.JUMP_UNI_STATE\
            and self.start_stop_flag==Constants.START and self.min_val_in_queue<1.25*self.start_stop_val:
                self.start_stop_flag = Constants.STOP
                self.start_stop_val = self.max_val_in_queue


            
        else:
            if self.minima_idx==self.frame_count and self.min_val_in_queue<1.25*self.start_stop_val\
                and self.start_stop_flag==Constants.START:
                self.start_stop_flag = Constants.STOP
                self.start_stop_val = self.max_val_in_queue
            


        if self.maxima_idx==self.frame_count:
            if self.max_val_in_queue/self.start_stop_val<max_thresh and self.start_stop_flag==Constants.START: # earlier 4
                self.start_stop_flag = Constants.STOP

        
        return self.start_stop_flag


    def process_action_state(self,key,frame_count,running_avg_queue, params):
        action_state,inc_counter, dec_counter,uni_counter,start_stop_flag ,start_stop_val,\
            maxima_idx ,minima_idx  = params
        self.update(running_avg_queue,action_state,
                             inc_counter, uni_counter, dec_counter,
                             start_stop_flag,frame_count,start_stop_val, 
                             maxima_idx,minima_idx)
        
        # slope execution running avg, identify min, max and diff_val
        self.get_max_min_diff_vals()

        # state of the queue inc, dec, uni
        self.identify_queue_state()

    
        # check minima
        self.check_minima()
        

        # check maxima
        self.check_maxima()
        

        # check threshold
        action_state = self.get_action_state(key)

        return action_state,self.inc_counter, self.dec_counter,self.uni_counter, self.start_stop_flag, self.start_stop_val,self.maxima_idx,self.minima_idx
    

    def update(self,running_avg_queue,action_state,
                             inc_counter, uni_counter, dec_counter,
                             start_stop_flag,frame_count,start_stop_val,
                             maxima_idx,minima_idx):
        self.running_avg_queue = running_avg_queue
        self.action_state = action_state
        self.inc_counter = inc_counter
        self.uni_counter = uni_counter
        self.dec_counter = dec_counter
        self.start_stop_flag = start_stop_flag
        self.frame_count = frame_count
        self.start_stop_val = start_stop_val
        self.maxima_idx = maxima_idx
        self.minima_idx = minima_idx