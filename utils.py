def intersects(obj, detection_zone):
  (b1_start_x, b1_start_y, b1_end_x, b1_end_y) = obj
  (b2_start_x, b2_start_y, b2_end_x, b2_end_y) = detection_zone

  max_start_x = max(b1_start_x, b2_start_x)
  min_end_x = min(b1_end_x, b2_end_x)
  min_end_y = min(b1_end_y, b2_end_y) 
  max_start_y = max(b1_start_y, b2_start_y)

  x_diff =  min_end_x - max_start_x  
  y_diff = min_end_y - max_start_y  

  if x_diff > 0 and y_diff > 0:
     return [max_start_x, max_start_y, min_end_x, min_end_y] 
  else:
     return False

