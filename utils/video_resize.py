import moviepy.editor as mp
clip = mp.VideoFileClip("../Recordings/Sample18_webcam/video_10min_move_webcam.mp4")
clip_resized = clip.resize(height=360)  # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
clip_resized.write_videofile("../Recordings/Sample18_webcam/resized_video_10min_move_webcam.mp4")#, codec='png')
